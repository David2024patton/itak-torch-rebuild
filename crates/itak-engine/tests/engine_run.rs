//! End-to-end tests of the engine binary's run path over a synthetic
//! GGUF file written by this test: real bytes on disk, parsed by the
//! `gguf` crate, loaded through the `itak-loader` seam, tokenized by
//! the real BPE tokenizer, and stepped through the real forward on
//! `cpu-compute`. The fixture is a tiny Glimmer (2 layers, hidden 32,
//! vocab 519) with correct tensor shapes and deterministic
//! pseudo-random weights, so the forward genuinely runs; the
//! assertions are about the pipeline (loads, tokenizes, generates,
//! stops for the configured reason), not about output quality, which
//! is the golden harness's job.

use std::path::PathBuf;

use gguf::MetadataValue;
use itak_engine::{EngineConfig, EngineError, StopReason};

// ---------- tiny GGUF writer (adapted from crates/itak-loader/tests) ----------

struct TestTensor {
    name: String,
    dims: Vec<u64>,
    data: Vec<u8>,
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
        out.extend_from_slice(&0u32.to_le_bytes()); // F32
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

// ---------- the tiny Glimmer fixture ----------

const HIDDEN: usize = 32;
const FFN: usize = 64;
const VOCAB: usize = 519;
const EOS_ID: u32 = 518;

/// Deterministic pseudo-random values in [-0.25, 0.25], identical on
/// every run and every machine.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let v = ((self.0 >> 40) & 0xFF_FFFF) as f32 / (1u64 << 24) as f32;
        (v - 0.5) * 0.5
    }
}

fn f32_tensor(rng: &mut Lcg, name: &str, dims: &[u64]) -> TestTensor {
    let n: u64 = dims.iter().product();
    let mut data = Vec::with_capacity(n as usize * 4);
    for _ in 0..n {
        data.extend_from_slice(&rng.next_f32().to_le_bytes());
    }
    TestTensor {
        name: name.to_string(),
        dims: dims.to_vec(),
        data,
    }
}

fn ones_tensor(name: &str, n: usize) -> TestTensor {
    let mut data = Vec::with_capacity(n * 4);
    for _ in 0..n {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }
    TestTensor {
        name: name.to_string(),
        dims: vec![n as u64],
        data,
    }
}

/// The tokenizer roundtrip fixture's vocabulary: every byte-mapped
/// character, every `<0xXX>` byte token, five merge products, an
/// unknown token, and one control token that serves as BOS and EOS.
fn vocab() -> (Vec<MetadataValue>, Vec<MetadataValue>, Vec<MetadataValue>) {
    let mut tokens: Vec<MetadataValue> = Vec::new();
    let mut types: Vec<MetadataValue> = Vec::new();
    for b in 0u8..=255 {
        tokens.push(MetadataValue::String(
            tokenizer::byte_to_char(b).to_string(),
        ));
        types.push(MetadataValue::Int32(1)); // Normal
    }
    for b in 0u8..=255 {
        tokens.push(MetadataValue::String(format!("<0x{b:02X}>")));
        types.push(MetadataValue::Int32(6)); // Byte
    }
    for merged in ["th", "the", "Ġthe", "he", "Ġa"] {
        tokens.push(MetadataValue::String(merged.to_string()));
        types.push(MetadataValue::Int32(1)); // Normal
    }
    tokens.push(MetadataValue::String("<unk>".to_string()));
    types.push(MetadataValue::Int32(2)); // Unknown
    tokens.push(MetadataValue::String("<|end_of_text|>".to_string()));
    types.push(MetadataValue::Int32(3)); // Control
    let merges = ["t h", "th e", "Ġ the", "h e", "Ġ a"]
        .iter()
        .map(|m| MetadataValue::String((*m).to_string()))
        .collect();
    (tokens, types, merges)
}

fn tiny_metadata() -> Vec<(String, MetadataValue)> {
    let s = |v: &str| MetadataValue::String(v.to_string());
    let u = MetadataValue::UInt32;
    let f = MetadataValue::Float32;
    let (tokens, types, merges) = vocab();
    vec![
        ("general.architecture".into(), s("muse-glimmer")),
        ("general.name".into(), s("Tiny Glimmer")),
        ("muse-glimmer.block_count".into(), u(2)),
        ("muse-glimmer.context_length".into(), u(256)),
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
        (
            "muse-glimmer.attention.sliding_window_pattern".into(),
            s("SG"),
        ),
        ("muse-glimmer.final_logit_softcapping".into(), f(20.0)),
        ("muse-glimmer.logit_scale".into(), f(1.0)),
        ("tokenizer.ggml.tokens".into(), MetadataValue::Array(tokens)),
        (
            "tokenizer.ggml.token_type".into(),
            MetadataValue::Array(types),
        ),
        ("tokenizer.ggml.merges".into(), MetadataValue::Array(merges)),
        ("tokenizer.ggml.bos_token_id".into(), u(EOS_ID)),
        ("tokenizer.ggml.eos_token_id".into(), u(EOS_ID)),
        (
            "tokenizer.ggml.add_bos_token".into(),
            MetadataValue::Bool(true),
        ),
    ]
}

fn tiny_tensors(omit: &[&str]) -> Vec<TestTensor> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let h = HIDDEN as u64;
    let mut out: Vec<TestTensor> = Vec::new();
    let mut push = |t: TestTensor| {
        if !omit.contains(&t.name.as_str()) {
            out.push(t);
        }
    };
    push(f32_tensor(
        &mut rng,
        "token_embd.weight",
        &[h, VOCAB as u64],
    ));
    push(ones_tensor("output_norm.weight", HIDDEN));
    push(f32_tensor(&mut rng, "output.weight", &[h, VOCAB as u64]));
    for layer in 0..2 {
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        push(ones_tensor(&name("attn_norm.weight"), HIDDEN));
        push(f32_tensor(&mut rng, &name("attn_q.weight"), &[h, 32]));
        push(f32_tensor(&mut rng, &name("attn_k.weight"), &[h, 16]));
        push(f32_tensor(&mut rng, &name("attn_v.weight"), &[h, 16]));
        push(ones_tensor(&name("attn_q_norm.weight"), 8));
        push(ones_tensor(&name("attn_k_norm.weight"), 8));
        push(f32_tensor(&mut rng, &name("attn_gate.weight"), &[h, 32]));
        push(f32_tensor(&mut rng, &name("attn_output.weight"), &[32, h]));
        push(ones_tensor(&name("post_attention_norm.weight"), HIDDEN));
        push(ones_tensor(&name("ffn_norm.weight"), HIDDEN));
        push(f32_tensor(
            &mut rng,
            &name("ffn_gate.weight"),
            &[h, FFN as u64],
        ));
        push(f32_tensor(
            &mut rng,
            &name("ffn_up.weight"),
            &[h, FFN as u64],
        ));
        push(f32_tensor(
            &mut rng,
            &name("ffn_down.weight"),
            &[FFN as u64, h],
        ));
        push(ones_tensor(&name("post_ffw_norm.weight"), HIDDEN));
    }
    out
}

fn write_fixture(tag: &str, omit: &[&str]) -> PathBuf {
    let md = tiny_metadata();
    let refs: Vec<(&str, MetadataValue)> =
        md.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let bytes = build(&refs, &tiny_tensors(omit));
    let path = std::env::temp_dir().join(format!(
        "itak-engine-test-{tag}-{}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("fixture writes");
    path
}

// ---------- the tests ----------

#[test]
fn engine_loads_tokenizes_and_generates() {
    let path = write_fixture("full", &[]);
    let mut cfg = EngineConfig::new(path.clone(), "The capital of France is".to_string());
    cfg.max_new_tokens = 4;
    let outcome = itak_engine::run(&cfg).expect("tiny model generates end to end");
    std::fs::remove_file(&path).ok();

    // BOS first (the file asks for it), then the tokenized prompt.
    assert_eq!(outcome.prompt_ids.first(), Some(&EOS_ID));
    assert!(
        outcome.prompt_ids.len() > 1,
        "prompt tokenized: {:?}",
        outcome.prompt_ids
    );
    // Every prompt id is inside the vocabulary the forward accepts.
    assert!(outcome.prompt_ids.iter().all(|&id| (id as usize) < VOCAB));

    assert!(
        !outcome.generated_ids.is_empty(),
        "at least the prefill pick is generated"
    );
    assert!(outcome.generated_ids.len() <= 4);
    assert_eq!(outcome.generated_ids.len(), outcome.generated_tokens.len());
    match outcome.stop_reason {
        StopReason::MaxTokens => assert_eq!(outcome.generated_ids.len(), 4),
        StopReason::Eos => {
            assert_eq!(outcome.generated_ids.last(), Some(&EOS_ID));
        }
        StopReason::ContextLimit => panic!("context cannot bind on this fixture"),
    }
}

#[test]
fn context_limit_refuses_an_overlong_prompt_before_loading() {
    let path = write_fixture("ctx", &[]);
    let mut cfg = EngineConfig::new(path.clone(), "The capital of France is".to_string());
    cfg.context_limit = Some(2);
    let err = itak_engine::run(&cfg).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(
        matches!(
            err,
            EngineError::PromptExceedsContext {
                context_limit: 2,
                ..
            }
        ),
        "got {err}"
    );
}

#[test]
fn missing_tensor_is_refused_by_name_through_the_loader_gate() {
    let path = write_fixture("missing", &["output.weight"]);
    let cfg = EngineConfig::new(path.clone(), "hello".to_string());
    let err = itak_engine::run(&cfg).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(
        matches!(err, EngineError::LoadWeights(_)),
        "coverage gate fires as a load refusal, got {err}"
    );
    assert!(
        err.to_string().contains("output.weight"),
        "the missing tensor is named: {err}"
    );
}

#[test]
fn context_limit_bounds_generation_when_it_binds_first() {
    let path = write_fixture("bound", &[]);
    let mut cfg = EngineConfig::new(path.clone(), "The capital of France is".to_string());
    cfg.max_new_tokens = 1;
    cfg.stop_on_eos = false;
    let probe = itak_engine::run(&cfg).expect("probe run");
    let prompt_len = probe.prompt_ids.len();
    // Room for exactly the prompt plus 2 generated tokens, with EOS
    // stopping off, so the context limit is the only thing that can
    // stop the run.
    cfg.max_new_tokens = 100;
    cfg.context_limit = Some(prompt_len + 2);
    let outcome = itak_engine::run(&cfg).expect("bounded run");
    std::fs::remove_file(&path).ok();
    assert_eq!(outcome.generated_ids.len(), 2);
    assert_eq!(outcome.stop_reason, StopReason::ContextLimit);
}
