//! Architecture dispatch tests (TODO T-2.8): synthetic GGUF files
//! whose `general.architecture` strings route through the arch
//! registry to their own modules, and a bogus string refused by
//! name with the supported list in the error.
//!
//! Files are real GGUF bytes built by the tiny writer below (the
//! same shape as the wiring tests), parsed by the real `gguf`
//! parser, and dispatched by [`ArchRegistry::standard`]. The
//! gemma2, olmo2, and deepseek2 fixtures carry only the metadata
//! their registered stub modules read, so a pass proves routing to
//! the right module, not a shared catch-all. The llama, qwen2, and
//! muse-glimmer fixtures carry full tiny configs and come back as
//! the real modules' config types.

use gguf::{GgufFile, MetadataValue};
use itak_loader::{
    select_engine, ArchEntry, ArchRegistry, Dispatched, EngineCandidate, Handler, LoaderError,
    ModuleKind,
};

// ---------- tiny GGUF writer (adapted from tests/wiring.rs) ----------

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

fn f32_tensor(name: &str, dims: &[u64], fill: f32) -> TestTensor {
    let count: u64 = dims.iter().product();
    let data: Vec<u8> = (0..count).flat_map(|_| fill.to_le_bytes()).collect();
    tensor(name, dims, 0, data)
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

fn refs(metadata: &[(String, MetadataValue)]) -> Vec<(&str, MetadataValue)> {
    metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect()
}

fn tokens(n: usize) -> MetadataValue {
    MetadataValue::Array(
        (0..n)
            .map(|i| MetadataValue::String(format!("t{i}")))
            .collect(),
    )
}

// ---------- fixtures ----------

/// Metadata for a registered-stub family: exactly the keys the stub
/// reads, under the family's own prefix.
fn stub_metadata(arch: &str) -> Vec<(String, MetadataValue)> {
    let p = |suffix: &str| format!("{arch}.{suffix}");
    vec![
        (
            "general.architecture".into(),
            MetadataValue::String(arch.into()),
        ),
        (p("block_count"), MetadataValue::UInt32(2)),
        (p("embedding_length"), MetadataValue::UInt32(32)),
        (p("context_length"), MetadataValue::UInt32(128)),
    ]
}

fn stub_file(arch: &str) -> GgufFile {
    let md = stub_metadata(arch);
    let tensors = [f32_tensor("token_embd.weight", &[4], 0.0)];
    GgufFile::from_bytes(build(&refs(&md), &tensors)).expect("stub fixture parses")
}

/// Metadata for a tiny full llama-family file (llama or qwen2):
/// hidden 32, FFN 48, 4 Q heads over 2 KV heads, vocab 64.
fn llama_family_metadata(arch: &str) -> Vec<(String, MetadataValue)> {
    let p = |suffix: &str| format!("{arch}.{suffix}");
    let mut entries = vec![
        (
            "general.architecture".into(),
            MetadataValue::String(arch.into()),
        ),
        (
            "general.name".into(),
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
        ("tokenizer.ggml.tokens".into(), tokens(64)),
    ];
    if arch == "llama" {
        entries.push((p("rope.dimension_count"), MetadataValue::UInt32(8)));
    }
    entries
}

fn llama_family_file(arch: &str, extra: &[TestTensor]) -> GgufFile {
    let md = llama_family_metadata(arch);
    GgufFile::from_bytes(build(&refs(&md), extra)).expect("llama-family fixture parses")
}

/// The tiny Glimmer metadata, same fixture as the wiring tests.
fn glimmer_metadata() -> Vec<(String, MetadataValue)> {
    let s = |v: &str| MetadataValue::String(v.to_string());
    let u = MetadataValue::UInt32;
    let f = MetadataValue::Float32;
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
        ("tokenizer.ggml.tokens".into(), tokens(128)),
    ]
}

// ---------- stub families route to their own modules ----------

#[test]
fn gemma2_routes_to_its_registered_module() {
    let registry = ArchRegistry::standard();
    let result = registry
        .dispatch(&stub_file("gemma2"))
        .expect("gemma2 routes");
    assert_eq!(result.entry.architecture(), "gemma2");
    assert_eq!(result.entry.kind(), ModuleKind::RegisteredStub);
    assert_eq!(result.entry.handler(), Handler::RegisteredStub);
    match &result.load {
        Dispatched::RegisteredStub(cfg) => {
            assert_eq!(cfg.architecture, "gemma2");
            assert_eq!(cfg.block_count, 2);
            assert_eq!(cfg.hidden, 32);
            assert_eq!(cfg.context_length, 128);
            assert_eq!(cfg.tensor_count, 1);
        }
        other => panic!("expected RegisteredStub load, got {other:?}"),
    }
}

#[test]
fn olmo2_routes_to_its_registered_module() {
    let registry = ArchRegistry::standard();
    let result = registry
        .dispatch(&stub_file("olmo2"))
        .expect("olmo2 routes");
    assert_eq!(result.entry.architecture(), "olmo2");
    match &result.load {
        Dispatched::RegisteredStub(cfg) => {
            assert_eq!(cfg.architecture, "olmo2");
            assert_eq!(cfg.block_count, 2);
            assert_eq!(cfg.hidden, 32);
        }
        other => panic!("expected RegisteredStub load, got {other:?}"),
    }
}

#[test]
fn deepseek2_routes_to_its_registered_module() {
    let registry = ArchRegistry::standard();
    let result = registry
        .dispatch(&stub_file("deepseek2"))
        .expect("deepseek2 routes");
    assert_eq!(result.entry.architecture(), "deepseek2");
    match &result.load {
        Dispatched::RegisteredStub(cfg) => {
            assert_eq!(cfg.architecture, "deepseek2");
            assert_eq!(cfg.context_length, 128);
        }
        other => panic!("expected RegisteredStub load, got {other:?}"),
    }
}

#[test]
fn stub_module_reads_its_own_prefix_not_another_familys() {
    // A gemma2-declared file carrying only llama-prefixed keys must
    // fail inside the gemma2 stub, naming the gemma2 key: routing
    // reached the right module and the module read its own keys.
    let mut md = stub_metadata("llama");
    for (k, v) in &mut md {
        if k == "general.architecture" {
            *v = MetadataValue::String("gemma2".into());
        }
    }
    let file = GgufFile::from_bytes(build(&refs(&md), &[])).expect("fixture parses");
    let err = ArchRegistry::standard()
        .dispatch(&file)
        .expect_err("gemma2 module cannot read llama keys");
    assert!(
        err.to_string().contains("gemma2.block_count"),
        "error names the gemma2 key: {err}"
    );
}

// ---------- full modules through the same path ----------

#[test]
fn llama_routes_to_the_full_llama_module() {
    let registry = ArchRegistry::standard();
    let result = registry
        .dispatch(&llama_family_file("llama", &[]))
        .expect("llama routes");
    assert_eq!(result.entry.kind(), ModuleKind::Full);
    assert_eq!(result.entry.handler(), Handler::LlamaFamily);
    match &result.load {
        Dispatched::LlamaFamily(cfg) => {
            assert_eq!(cfg.architecture, "llama");
            assert_eq!(cfg.block_count, 2);
            assert_eq!(cfg.hidden, 32);
            assert_eq!(cfg.head_dim, 8);
            assert_eq!(cfg.rope_dim, 8);
        }
        other => panic!("expected LlamaFamily load, got {other:?}"),
    }
}

#[test]
fn qwen2_routes_to_the_full_llama_module() {
    let registry = ArchRegistry::standard();
    let result = registry
        .dispatch(&llama_family_file("qwen2", &[]))
        .expect("qwen2 routes");
    match &result.load {
        Dispatched::LlamaFamily(cfg) => {
            assert_eq!(cfg.architecture, "qwen2");
            assert!(cfg.attention_biases_required());
        }
        other => panic!("expected LlamaFamily load, got {other:?}"),
    }
}

#[test]
fn muse_glimmer_routes_to_the_full_glimmer_module() {
    let md = glimmer_metadata();
    let file = GgufFile::from_bytes(build(&refs(&md), &[])).expect("glimmer fixture parses");
    let result = ArchRegistry::standard()
        .dispatch(&file)
        .expect("muse-glimmer routes");
    assert_eq!(result.entry.architecture(), "muse-glimmer");
    assert_eq!(result.entry.handler(), Handler::Glimmer);
    match &result.load {
        Dispatched::Glimmer(cfg) => {
            assert_eq!(cfg.block_count, 2);
            assert_eq!(cfg.hidden, 32);
            assert_eq!(cfg.vocab_size, 128);
        }
        other => panic!("expected Glimmer load, got {other:?}"),
    }
}

// ---------- refusals ----------

#[test]
fn bogus_architecture_is_refused_with_the_supported_list() {
    let md = vec![(
        "general.architecture".to_string(),
        MetadataValue::String("not-a-real-arch".into()),
    )];
    let file = GgufFile::from_bytes(build(&refs(&md), &[])).expect("fixture parses");
    let err = ArchRegistry::standard()
        .dispatch(&file)
        .expect_err("bogus arch must be refused");
    match &err {
        LoaderError::UnknownArchitecture { found, supported } => {
            assert_eq!(found, "not-a-real-arch");
            assert_eq!(
                supported,
                &[
                    "deepseek2",
                    "gemma2",
                    "llama",
                    "muse-glimmer",
                    "olmo2",
                    "qwen2"
                ]
            );
        }
        other => panic!("expected UnknownArchitecture, got {other:?}"),
    }
    let text = err.to_string();
    assert!(
        text.contains("not-a-real-arch"),
        "error names the string: {text}"
    );
    for name in [
        "deepseek2",
        "gemma2",
        "llama",
        "muse-glimmer",
        "olmo2",
        "qwen2",
    ] {
        assert!(
            text.contains(name),
            "error lists supported arch {name}: {text}"
        );
    }
}

#[test]
fn missing_architecture_key_is_refused_by_key_name() {
    let file = GgufFile::from_bytes(build(&[], &[])).expect("fixture parses");
    let err = ArchRegistry::standard()
        .dispatch(&file)
        .expect_err("a file that declares nothing cannot load");
    assert!(
        err.to_string().contains("general.architecture"),
        "error names the key: {err}"
    );
}

#[test]
fn runtime_registered_family_routes_through_the_same_registry() {
    // The registry is the mechanism, not a hardcoded match: a
    // family registered at runtime routes the same way.
    let mut registry = ArchRegistry::new();
    registry
        .register(ArchEntry::stub("phi3", 1))
        .expect("register phi3");
    let result = registry.dispatch(&stub_file("phi3")).expect("phi3 routes");
    assert_eq!(result.load.architecture(), "phi3");
    // And the standard entries are absent from this registry.
    let err = registry
        .dispatch(&stub_file("gemma2"))
        .expect_err("gemma2 is not registered here");
    assert!(matches!(err, LoaderError::UnknownArchitecture { .. }));
    assert!(
        err.to_string().contains("phi3"),
        "supported list reflects this registry: {err}"
    );
}

// ---------- engine selection is by capability, never file size ----------

#[test]
fn engine_selection_is_identical_for_small_and_large_files_of_one_arch() {
    let registry = ArchRegistry::standard();
    let small = registry
        .dispatch(&llama_family_file("llama", &[]))
        .expect("small llama routes");
    // Same architecture, same config, plus a 4 MB filler tensor:
    // nothing about the file's size may change the engine choice.
    let big = registry
        .dispatch(&llama_family_file(
            "llama",
            &[f32_tensor("filler.weight", &[1_000_000], 0.0)],
        ))
        .expect("large llama routes");
    let candidates = [
        EngineCandidate {
            name: "vulkan-7900xtx",
            available: false,
            forward_architectures: &["llama", "qwen2", "muse-glimmer"],
        },
        EngineCandidate {
            name: "cpu-reference",
            available: true,
            forward_architectures: &["llama", "qwen2", "muse-glimmer"],
        },
    ];
    let small_engine = select_engine(&small, &candidates).expect("small selects");
    let big_engine = select_engine(&big, &candidates).expect("large selects");
    assert_eq!(small_engine, big_engine);
    assert_eq!(small_engine.engine, "cpu-reference");
}

#[test]
fn tiny_stub_file_gets_no_forward_engine() {
    // One tensor, a few hundred bytes: if selection routed by size
    // this is exactly the file that would slip onto an engine. It
    // is refused because the gemma2 module provides no forward.
    let registry = ArchRegistry::standard();
    let result = registry
        .dispatch(&stub_file("gemma2"))
        .expect("gemma2 routes");
    let candidates = [EngineCandidate {
        name: "cpu-reference",
        available: true,
        forward_architectures: &["gemma2", "llama"],
    }];
    let err = select_engine(&result, &candidates).expect_err("stub has no forward");
    assert!(matches!(err, LoaderError::NoCapableEngine { .. }));
    assert!(err.to_string().contains("gemma2"), "error: {err}");
}
