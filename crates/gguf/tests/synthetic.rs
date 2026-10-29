//! End-to-end tests over synthetic GGUF files built by a tiny writer in
//! this module. Dequantization expectations are hand-computed from the
//! ggml block layouts, not produced by the code under test.

use gguf::{GgufError, GgufFile, MetadataValue};

// ---------- tiny GGUF writer ----------

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

/// Build a complete GGUF v3 file. Tensor data offsets are aligned the way
/// real writers align them.
fn build(metadata: &[(&str, MetadataValue)], tensors: &[TestTensor], alignment: u64) -> Vec<u8> {
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

fn f16le(bits: u16) -> [u8; 2] {
    bits.to_le_bytes()
}

const F16_ONE: u16 = 0x3c00;
const F16_TWO: u16 = 0x4000;
const F16_HALF: u16 = 0x3800;

// ---------- header and metadata ----------

#[test]
fn header_and_metadata_parse() {
    let bytes = build(
        &[
            (
                "general.architecture",
                MetadataValue::String("testarch".into()),
            ),
            ("general.name", MetadataValue::String("Tiny Test".into())),
            ("test.count", MetadataValue::UInt32(42)),
            ("test.delta", MetadataValue::Int32(-7)),
            ("test.ratio", MetadataValue::Float32(0.25)),
            ("test.flag", MetadataValue::Bool(true)),
            (
                "test.tags",
                MetadataValue::Array(vec![
                    MetadataValue::String("a".into()),
                    MetadataValue::String("bc".into()),
                ]),
            ),
        ],
        &[],
        32,
    );
    let file = GgufFile::from_bytes(bytes).unwrap();
    assert_eq!(file.version(), 3);
    assert_eq!(file.architecture().unwrap(), "testarch");
    let md = file.metadata();
    assert_eq!(md.get_string("general.name").unwrap(), "Tiny Test");
    assert_eq!(md.get_u32("test.count").unwrap(), 42);
    assert_eq!(md.get_i64("test.delta").unwrap(), -7);
    assert_eq!(md.get_f32("test.ratio").unwrap(), 0.25);
    assert!(md.get_bool("test.flag").unwrap());
    assert_eq!(
        md.get_string_array("test.tags").unwrap(),
        vec!["a".to_string(), "bc".to_string()]
    );
}

#[test]
fn missing_and_mistyped_metadata_are_named_errors() {
    let bytes = build(
        &[("general.architecture", MetadataValue::String("x".into()))],
        &[],
        32,
    );
    let file = GgufFile::from_bytes(bytes).unwrap();
    let err = file.metadata().get_u32("nope.key").unwrap_err();
    assert!(matches!(err, GgufError::MissingMetadata(k) if k == "nope.key"));
    let err = file.metadata().get_u32("general.architecture").unwrap_err();
    assert!(matches!(err, GgufError::MetadataTypeMismatch { .. }));
}

#[test]
fn bad_magic_and_wrong_version_are_rejected() {
    let mut bad = build(&[], &[], 32);
    bad[0] = b'X';
    assert!(matches!(
        GgufFile::from_bytes(bad),
        Err(GgufError::BadMagic { .. })
    ));
    let mut v2 = build(&[], &[], 32);
    v2[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(
        GgufFile::from_bytes(v2),
        Err(GgufError::UnsupportedVersion(2))
    ));
}

#[test]
fn truncated_file_fails() {
    let bytes = build(
        &[("general.architecture", MetadataValue::String("x".into()))],
        &[tensor("w", &[4], 0, vec![0u8; 16])],
        32,
    );
    let cut = bytes[..bytes.len() / 2].to_vec();
    assert!(GgufFile::from_bytes(cut).is_err());
}

#[test]
fn tensor_data_past_eof_is_rejected_at_parse() {
    let mut bytes = build(&[], &[tensor("w", &[4], 0, vec![0u8; 16])], 32);
    bytes.truncate(bytes.len() - 10);
    let err = GgufFile::from_bytes(bytes).unwrap_err();
    assert!(matches!(err, GgufError::TensorOutOfBounds { .. }));
}

// ---------- model config ----------

fn config_metadata(with_kv: bool) -> Vec<(String, MetadataValue)> {
    let mut v: Vec<(String, MetadataValue)> = vec![
        (
            "general.architecture".into(),
            MetadataValue::String("testarch".into()),
        ),
        ("testarch.block_count".into(), MetadataValue::UInt32(4)),
        (
            "testarch.embedding_length".into(),
            MetadataValue::UInt32(128),
        ),
        (
            "testarch.feed_forward_length".into(),
            MetadataValue::UInt32(256),
        ),
        (
            "testarch.attention.head_count".into(),
            MetadataValue::UInt32(8),
        ),
        (
            "testarch.context_length".into(),
            MetadataValue::UInt32(2048),
        ),
        (
            "testarch.rope.freq_base".into(),
            MetadataValue::Float32(500_000.0),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            MetadataValue::Array(vec![
                MetadataValue::String("<unk>".into()),
                MetadataValue::String("a".into()),
                MetadataValue::String("b".into()),
            ]),
        ),
        (
            "tokenizer.ggml.bos_token_id".into(),
            MetadataValue::UInt32(0),
        ),
        (
            "tokenizer.ggml.eos_token_id".into(),
            MetadataValue::UInt32(1),
        ),
    ];
    if with_kv {
        v.push((
            "testarch.attention.head_count_kv".into(),
            MetadataValue::UInt32(2),
        ));
    }
    v
}

#[test]
fn model_config_reads_the_files_own_keys() {
    let owned = config_metadata(true);
    let refs: Vec<(&str, MetadataValue)> =
        owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let file = GgufFile::from_bytes(build(&refs, &[], 32)).unwrap();
    let cfg = file.model_config().unwrap();
    assert_eq!(cfg.architecture, "testarch");
    assert_eq!(cfg.layer_count, 4);
    assert_eq!(cfg.hidden_size, 128);
    assert_eq!(cfg.ffn_size, Some(256));
    assert_eq!(cfg.head_count, 8);
    assert_eq!(cfg.head_count_kv, 2);
    assert_eq!(cfg.context_length, 2048);
    assert_eq!(cfg.rope_base, Some(500_000.0));
    assert_eq!(cfg.vocab_size, Some(3));
    let tok = file.tokenizer().unwrap();
    assert_eq!(tok.tokens.len(), 3);
    assert_eq!(tok.bos_token_id, Some(0));
    assert_eq!(tok.eos_token_id, Some(1));
}

#[test]
fn kv_head_count_falls_back_to_query_heads() {
    let owned = config_metadata(false);
    let refs: Vec<(&str, MetadataValue)> =
        owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let file = GgufFile::from_bytes(build(&refs, &[], 32)).unwrap();
    let cfg = file.model_config().unwrap();
    assert_eq!(cfg.head_count_kv, cfg.head_count);
}

#[test]
fn per_layer_kv_head_counts_from_hybrid_files() {
    // Real shape from a granitehybrid GGUF: head_count_kv is an array with
    // the KV count on attention layers and 0 on SSM layers.
    let mut owned = config_metadata(false);
    owned.push((
        "testarch.attention.head_count_kv".into(),
        MetadataValue::Array(vec![
            MetadataValue::Int32(0),
            MetadataValue::Int32(0),
            MetadataValue::Int32(4),
            MetadataValue::Int32(0),
        ]),
    ));
    let refs: Vec<(&str, MetadataValue)> =
        owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let file = GgufFile::from_bytes(build(&refs, &[], 32)).unwrap();
    let cfg = file.model_config().unwrap();
    assert_eq!(cfg.head_count_kv, 4);
    assert_eq!(cfg.head_count_kv_per_layer, Some(vec![0, 0, 4, 0]));
    assert_eq!(cfg.head_count_per_layer, None);
}

#[test]
fn alignment_comes_from_metadata() {
    let bytes = build(
        &[
            ("general.architecture", MetadataValue::String("x".into())),
            ("general.alignment", MetadataValue::UInt32(64)),
        ],
        &[tensor("w", &[2], 0, vec![0u8; 8])],
        64,
    );
    let file = GgufFile::from_bytes(bytes).unwrap();
    assert_eq!(file.alignment(), 64);
    assert_eq!(file.data_start() % 64, 0);
    assert_eq!(file.load_f32("w").unwrap().len(), 2);
}

// ---------- dequantization, hand-computed ----------

#[test]
fn dequant_f32_f16_bf16() {
    let f32_data: Vec<u8> = [1.5f32, -0.25]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let f16_data: Vec<u8> = [F16_ONE, 0xbc00, F16_HALF]
        .iter()
        .flat_map(|b| f16le(*b))
        .collect();
    // BF16: 1.0 = 0x3f80, -2.5 = 0xc020.
    let bf16_data: Vec<u8> = [0x3f80u16, 0xc020]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let file = GgufFile::from_bytes(build(
        &[],
        &[
            tensor("f32w", &[2], 0, f32_data),
            tensor("f16w", &[3], 1, f16_data),
            tensor("bf16w", &[2], 30, bf16_data),
        ],
        32,
    ))
    .unwrap();
    assert_eq!(file.load_f32("f32w").unwrap(), vec![1.5, -0.25]);
    assert_eq!(file.load_f32("f16w").unwrap(), vec![1.0, -1.0, 0.5]);
    assert_eq!(file.load_f32("bf16w").unwrap(), vec![1.0, -2.5]);
}

#[test]
fn bf16_id_30_is_not_i2_s() {
    // Regression for the old loader's fatal collision: mainline id 30 is
    // BF16. If this ever decodes as anything else, the test goes red.
    let data: Vec<u8> = [0x3f80u16].iter().flat_map(|b| b.to_le_bytes()).collect();
    let file = GgufFile::from_bytes(build(&[], &[tensor("w", &[1], 30, data)], 32)).unwrap();
    assert_eq!(file.tensor_info("w").unwrap().type_name(), "BF16");
    assert_eq!(file.load_f32("w").unwrap(), vec![1.0]);
}

#[test]
fn dequant_q8_0() {
    let mut data = Vec::new();
    data.extend_from_slice(&f16le(F16_HALF)); // d = 0.5
    let mut qs = [0i8; 32];
    qs[0] = 2;
    qs[1] = -4;
    qs[31] = 100;
    data.extend(qs.iter().map(|q| *q as u8));
    let file = GgufFile::from_bytes(build(&[], &[tensor("w", &[32], 8, data)], 32)).unwrap();
    let out = file.load_f32("w").unwrap();
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], -2.0);
    assert_eq!(out[31], 50.0);
    assert!(out[2..31].iter().all(|v| *v == 0.0));
}

#[test]
fn dequant_q4_k() {
    // Hand-computed: d = 2.0, dmin = 0.5.
    // Sub-block 0: scale 3, min 2 -> factor 6.0, offset 1.0.
    // Sub-block 1: scale 1, min 0 -> factor 2.0, offset 0.0.
    let mut block = Vec::new();
    block.extend_from_slice(&f16le(F16_TWO));
    block.extend_from_slice(&f16le(F16_HALF));
    let mut scales = [0u8; 12];
    scales[0] = 3;
    scales[4] = 2;
    scales[1] = 1;
    scales[5] = 0;
    block.extend_from_slice(&scales);
    let mut qs = [0u8; 128];
    qs[0] = 0x21;
    qs[1] = 0x43;
    block.extend_from_slice(&qs);
    assert_eq!(block.len(), 144);
    let file = GgufFile::from_bytes(build(&[], &[tensor("w", &[256], 12, block)], 32)).unwrap();
    let out = file.load_f32("w").unwrap();
    assert_eq!(out[0], 5.0); // 6*1 - 1
    assert_eq!(out[1], 17.0); // 6*3 - 1
    assert_eq!(out[2], -1.0); // 6*0 - 1
    assert_eq!(out[32], 4.0); // 2*2 - 0 (high nibble of qs[0])
    assert_eq!(out[33], 8.0); // 2*4 - 0 (high nibble of qs[1])
    assert_eq!(out[64], 0.0); // sub-block 2 has zero scale and min
}

#[test]
fn dequant_q5_k() {
    // Hand-computed: d = 1.0, dmin = 0. Sub-block 0 scale 2, sub-block 1
    // scale 1. ql[0] = 0x53, qh[0] = 0x01 sets element 0's fifth bit.
    let mut block = Vec::new();
    block.extend_from_slice(&f16le(F16_ONE));
    block.extend_from_slice(&f16le(0));
    let mut scales = [0u8; 12];
    scales[0] = 2;
    scales[1] = 1;
    block.extend_from_slice(&scales);
    let mut qh = [0u8; 32];
    qh[0] = 0x01;
    block.extend_from_slice(&qh);
    let mut ql = [0u8; 128];
    ql[0] = 0x53;
    block.extend_from_slice(&ql);
    assert_eq!(block.len(), 176);
    let file = GgufFile::from_bytes(build(&[], &[tensor("w", &[256], 13, block)], 32)).unwrap();
    let out = file.load_f32("w").unwrap();
    assert_eq!(out[0], 38.0); // 2 * (3 + 16)
    assert_eq!(out[32], 5.0); // 1 * 5 (high nibble, no fifth bit)
    assert_eq!(out[1], 0.0);
}

#[test]
fn dequant_q6_k() {
    // Hand-computed: d = 1.0. Scales: sc[0]=2, sc[2]=1, sc[4]=3, sc[6]=1.
    // ql[0] = 0x23, ql[32] = 0x05, qh[0] = 0x41 (bit 0 for stream 1,
    // bit 6 for stream 4).
    let mut block = vec![0u8; 210];
    block[0] = 0x23; // ql[0]
    block[32] = 0x05; // ql[32]
    block[128] = 0x41; // qh[0]
    block[192] = 2; // sc[0]
    block[194] = 1; // sc[2]
    block[196] = 3; // sc[4]
    block[198] = 1; // sc[6]
    block[208..210].copy_from_slice(&f16le(F16_ONE));
    let file = GgufFile::from_bytes(build(&[], &[tensor("w", &[256], 14, block)], 32)).unwrap();
    let out = file.load_f32("w").unwrap();
    assert_eq!(out[0], -26.0); // sc0 * ((3 | 1<<4) - 32) = 2 * -13
    assert_eq!(out[32], -27.0); // sc2 * (5 - 32)
    assert_eq!(out[64], -90.0); // sc4 * (2 - 32) = 3 * -30
    assert_eq!(out[96], -16.0); // sc6 * ((0 | 1<<4) - 32) = 1 * -16
}

#[test]
fn two_dim_tensor_directory_and_load() {
    let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let file = GgufFile::from_bytes(build(&[], &[tensor("w", &[4, 2], 0, data)], 32)).unwrap();
    let info = file.tensor_info("w").unwrap();
    assert_eq!(info.dims(), &[4, 2]);
    assert_eq!(info.element_count(), 8);
    let out = file.load_f32("w").unwrap();
    assert_eq!(out, (0..8).map(|i| i as f32).collect::<Vec<_>>());
}

// ---------- fail-loud behavior ----------

#[test]
fn missing_tensor_is_a_hard_error() {
    let file =
        GgufFile::from_bytes(build(&[], &[tensor("a.weight", &[1], 0, vec![0; 4])], 32)).unwrap();
    let err = file.load_f32("b.weight").unwrap_err();
    assert!(matches!(&err, GgufError::MissingTensor(n) if n == "b.weight"));
    assert!(err.to_string().contains("b.weight"));
}

#[test]
fn unsupported_types_are_hard_errors_that_name_the_type() {
    // Q3_K (11) and Q8_1 (9): the readers the old Go loader lacked. They
    // must fail loudly here, never decode as something else.
    for (type_id, name) in [(11u32, "Q3_K"), (9, "Q8_1")] {
        let file = GgufFile::from_bytes(build(
            &[],
            &[tensor("w", &[256], type_id, vec![0u8; 64])],
            32,
        ))
        .unwrap();
        let err = file.load_f32("w").unwrap_err();
        match err {
            GgufError::UnsupportedTensorType {
                ref tensor,
                type_id: id,
                ref type_name,
            } => {
                assert_eq!(tensor, "w");
                assert_eq!(id, type_id);
                assert_eq!(type_name, name);
            }
            other => panic!("expected UnsupportedTensorType, got {other:?}"),
        }
        assert!(err.to_string().contains(name));
    }
    // An id mainline ggml has not assigned is named by number.
    let file =
        GgufFile::from_bytes(build(&[], &[tensor("w", &[8], 99, vec![0u8; 32])], 32)).unwrap();
    let err = file.load_f32("w").unwrap_err();
    assert!(err.to_string().contains("99"));
}

#[test]
fn bad_element_count_is_rejected_at_parse() {
    // 100 elements is not a whole number of 256-element Q4_K blocks.
    let err = GgufFile::from_bytes(build(&[], &[tensor("w", &[100], 12, vec![0u8; 144])], 32))
        .unwrap_err();
    assert!(matches!(err, GgufError::BadElementCount { .. }));
}
