//! End-to-end wiring tests over synthetic GGUF files: real bytes in,
//! config, tensors, and coverage out, through the same adapters the
//! engine will use. The tiny GGUF writer is adapted from the `gguf`
//! crate's synthetic tests; expectations here are the muse-glimmer
//! module's measured inventory, not the writer's opinion.

use gguf::{GgufFile, MetadataValue};
use itak_loader::{coverage, load_config, load_weights, GgufTensorSource, LoaderError};
use muse_glimmer::{Dtype, GlimmerError, TensorSource};

// ---------- tiny GGUF writer (adapted from crates/gguf/tests) ----------

struct TestTensor {
    name: String,
    dims: Vec<u64>,
    type_id: u32,
    data: Vec<u8>,
}

fn tensor(name: &str, dims: &[u64], type_id: u32, data: Vec<u8>) -> TestTensor {
    TestTensor {
        name: name.to_string(),
        dims: dims.to_vec(),
        type_id,
        data,
    }
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
        MetadataValue::Array(_) => panic!("nested arrays are not valid GGUF"),
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
        _ => write_scalar(out, v),
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

fn build(metadata: &[(&str, MetadataValue)], tensors: &[TestTensor]) -> Vec<u8> {
    let alignment = 32u64;
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        write_string(&mut out, k);
        write_value(&mut out, v);
    }
    let mut offsets = Vec::new();
    let mut cursor = 0u64;
    for t in tensors {
        let aligned = align_up(cursor, alignment);
        offsets.push(aligned);
        cursor = aligned + t.data.len() as u64;
    }
    for (t, off) in tensors.iter().zip(&offsets) {
        write_string(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.type_id.to_le_bytes());
        out.extend_from_slice(&off.to_le_bytes());
    }
    let data_start = align_up(out.len() as u64, alignment);
    out.resize(data_start as usize, 0);
    for (t, off) in tensors.iter().zip(&offsets) {
        out.resize((data_start + off) as usize, 0);
        out.extend_from_slice(&t.data);
    }
    out
}

// ---------- tiny Glimmer fixture ----------

/// The 14 per-layer suffixes of the measured inventory.
const SUFFIXES: [&str; 14] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "attn_gate.weight",
    "post_attention_norm.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "post_ffw_norm.weight",
];

fn tiny_metadata() -> Vec<(String, MetadataValue)> {
    let s = |v: &str| MetadataValue::String(v.to_string());
    let u = MetadataValue::UInt32;
    let f = MetadataValue::Float32;
    let tokens: Vec<MetadataValue> = (0..128).map(|i| s(&format!("t{i}"))).collect();
    vec![
        ("general.architecture".into(), s("muse-glimmer")),
        ("general.name".into(), s("Tiny Glimmer")),
        ("muse-glimmer.block_count".into(), u(2)),
        ("muse-glimmer.context_length".into(), u(128)),
        ("muse-glimmer.embedding_length".into(), u(32)),
        ("muse-glimmer.feed_forward_length".into(), u(64)),
        ("muse-glimmer.attention.head_count".into(), u(4)),
        ("muse-glimmer.attention.head_count_kv".into(), u(2)),
        ("muse-glimmer.attention.key_length".into(), u(8)),
        ("muse-glimmer.attention.value_length".into(), u(8)),
        ("muse-glimmer.rope.freq_base".into(), f(10_000.0)),
        (
            "muse-glimmer.attention.layer_norm_rms_epsilon".into(),
            f(1e-6),
        ),
        ("muse-glimmer.attention.sliding_window".into(), u(16)),
        ("muse-glimmer.final_logit_softcapping".into(), f(20.0)),
        ("muse-glimmer.logit_scale".into(), f(1.0)),
        ("tokenizer.ggml.tokens".into(), MetadataValue::Array(tokens)),
    ]
}

fn refs(metadata: &[(String, MetadataValue)]) -> Vec<(&str, MetadataValue)> {
    metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect()
}

/// Every expected tensor of the tiny model. All F32 `[4]` except one
/// Q4_K tensor, so dtype mapping is exercised on the serve path.
fn tiny_tensors() -> Vec<TestTensor> {
    let mut out = vec![
        tensor("token_embd.weight", &[4], 0, vec![0u8; 16]),
        tensor("output_norm.weight", &[4], 0, vec![0u8; 16]),
        tensor("output.weight", &[4], 0, vec![0u8; 16]),
    ];
    for layer in 0..2 {
        for suffix in SUFFIXES {
            let name = format!("blk.{layer}.{suffix}");
            if name == "blk.0.attn_q.weight" {
                out.push(tensor(&name, &[256], 12, vec![0u8; 144]));
            } else {
                out.push(tensor(&name, &[4], 0, vec![0u8; 16]));
            }
        }
    }
    out
}

fn tiny_file() -> GgufFile {
    let md = tiny_metadata();
    GgufFile::from_bytes(build(&refs(&md), &tiny_tensors())).expect("tiny file parses")
}

// ---------- the wiring ----------

#[test]
fn config_loads_end_to_end_from_gguf_bytes() {
    let file = tiny_file();
    let cfg = load_config(&file).expect("config loads");
    assert_eq!(cfg.block_count, 2);
    assert_eq!(cfg.hidden, 32);
    assert_eq!(cfg.ffn_hidden, 64);
    assert_eq!(cfg.q_heads, 4);
    assert_eq!(cfg.kv_heads, 2);
    assert_eq!(cfg.gqa_group(), 2);
    assert_eq!(cfg.vocab_size, 128);
    assert_eq!(cfg.name.as_deref(), Some("Tiny Glimmer"));
}

#[test]
fn metadata_adapter_coerces_stored_integer_widths() {
    // The real world stores counts at whatever width the writer chose;
    // the trait contract is coercion, and the gguf getter owns it.
    let mut md = tiny_metadata();
    for (k, v) in &mut md {
        if k == "muse-glimmer.block_count" {
            *v = MetadataValue::UInt64(2);
        }
        if k == "muse-glimmer.rope.freq_base" {
            *v = MetadataValue::Float64(10_000.0);
        }
    }
    let file = GgufFile::from_bytes(build(&refs(&md), &tiny_tensors())).unwrap();
    let cfg = load_config(&file).expect("wide storage still parses");
    assert_eq!(cfg.block_count, 2);
    assert!((cfg.rope_freq_base - 10_000.0).abs() < 0.01);
}

#[test]
fn tensor_source_serves_measured_facts() {
    let file = tiny_file();
    let source = GgufTensorSource::new(&file);
    let q = source.tensor("blk.0.attn_q.weight").expect("q serves");
    assert_eq!(q.dtype, Dtype::Q4K);
    assert_eq!(q.shape, vec![256]);
    assert_eq!(q.data.len(), 144);
    assert_eq!(q.elements(), 256);
    let norm = source.tensor("output_norm.weight").expect("norm serves");
    assert_eq!(norm.dtype, Dtype::F32);
    assert_eq!(norm.data.len(), 16);
}

#[test]
fn missing_tensor_is_named_missing() {
    let file = tiny_file();
    let source = GgufTensorSource::new(&file);
    let err = source.tensor("blk.9.ffn_down.weight").unwrap_err();
    match err {
        GlimmerError::MissingTensor { name } => assert_eq!(name, "blk.9.ffn_down.weight"),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn unsupported_type_is_unreadable_not_missing() {
    // Q3_K (mainline id 11): present in the directory, but the gguf
    // crate cannot size or read it. The seam must say unreadable, with
    // the type named, never missing and never placeholder bytes.
    let mut tensors = tiny_tensors();
    for t in &mut tensors {
        if t.name == "blk.0.attn_q.weight" {
            t.type_id = 11;
            t.data = vec![0u8; 64];
        }
    }
    let md = tiny_metadata();
    let file = GgufFile::from_bytes(build(&refs(&md), &tensors)).unwrap();
    let source = GgufTensorSource::new(&file);
    let err = source.tensor("blk.0.attn_q.weight").unwrap_err();
    match err {
        GlimmerError::TensorUnreadable { name, detail } => {
            assert_eq!(name, "blk.0.attn_q.weight");
            assert!(detail.contains("Q3_K"), "detail names the type: {detail}");
        }
        other => panic!("expected TensorUnreadable, got {other:?}"),
    }
}

#[test]
fn coverage_complete_and_weights_load() {
    let file = tiny_file();
    let cfg = load_config(&file).unwrap();
    let report = coverage(&file, &cfg);
    assert_eq!(report.expected_count(), 31);
    assert_eq!(report.found_count(), 31);
    assert!(report.is_complete());
    assert!(report.unexpected().is_empty());
    report.require_complete().expect("complete coverage passes");
    let weights = load_weights(&file, &cfg).expect("weights load");
    assert_eq!(weights.layers.len(), 2);
    assert_eq!(weights.output.name, "output.weight");
    assert_eq!(weights.final_norm.name, "output_norm.weight");
    assert_eq!(weights.layers[0].attn_q.dtype, Dtype::Q4K);
}

#[test]
fn coverage_names_every_missing_tensor() {
    let omit = ["output.weight", "blk.1.ffn_down.weight"];
    let tensors: Vec<TestTensor> = tiny_tensors()
        .into_iter()
        .filter(|t| !omit.contains(&t.name.as_str()))
        .collect();
    let md = tiny_metadata();
    let file = GgufFile::from_bytes(build(&refs(&md), &tensors)).unwrap();
    let cfg = load_config(&file).unwrap();
    let report = coverage(&file, &cfg);
    assert_eq!(report.found_count(), 29);
    assert_eq!(
        report.missing(),
        ["output.weight", "blk.1.ffn_down.weight"].as_slice()
    );
    let err = report.require_complete().unwrap_err();
    assert!(err.to_string().contains("output.weight"));
    assert!(err.to_string().contains("blk.1.ffn_down.weight"));
    // load_weights is gated on the same check and fails the same way.
    let err = load_weights(&file, &cfg).unwrap_err();
    assert!(matches!(err, LoaderError::MissingTensors(_)));
    assert!(err.to_string().contains("blk.1.ffn_down.weight"));
}

#[test]
fn unexpected_tensors_are_reported_not_fatal() {
    let mut tensors = tiny_tensors();
    tensors.push(tensor("vision_tower.weight", &[4], 0, vec![0u8; 16]));
    let md = tiny_metadata();
    let file = GgufFile::from_bytes(build(&refs(&md), &tensors)).unwrap();
    let cfg = load_config(&file).unwrap();
    let report = coverage(&file, &cfg);
    assert!(report.is_complete());
    assert_eq!(report.unexpected(), ["vision_tower.weight"].as_slice());
}

#[test]
fn wrong_architecture_is_refused() {
    let mut md = tiny_metadata();
    for (k, v) in &mut md {
        if k == "general.architecture" {
            *v = MetadataValue::String("llama".to_string());
        }
    }
    let file = GgufFile::from_bytes(build(&refs(&md), &tiny_tensors())).unwrap();
    let err = load_config(&file).unwrap_err();
    match err {
        GlimmerError::WrongArchitecture { found } => assert_eq!(found, "llama"),
        other => panic!("expected WrongArchitecture, got {other:?}"),
    }
}
