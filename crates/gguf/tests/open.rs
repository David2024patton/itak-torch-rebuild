//! `GgufFile::open` (memory-mapped, PLAN T-2.1) must be
//! indistinguishable from `GgufFile::from_bytes` (owned buffer) on
//! every observable: parsed metadata, tensor directory, raw tensor
//! bytes, dequantized values, and the named error for every failure
//! the in-memory path names. The tiny GGUF writer is adapted from
//! `tests/synthetic.rs`.

use std::path::PathBuf;

use gguf::{GgufError, GgufFile, MetadataValue};

// ---------- tiny GGUF writer (adapted from tests/synthetic.rs) ----------

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
        MetadataValue::UInt32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::String(s) => write_string(out, s),
        _ => panic!("this writer only carries the value kinds these tests use"),
    }
}

fn write_value(out: &mut Vec<u8>, v: &MetadataValue) {
    out.extend_from_slice(&v.type_id().to_le_bytes());
    write_scalar(out, v);
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

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

// ---------- temp file handling ----------

/// A temp file that removes itself on drop, so a failing assert does
/// not litter the temp directory.
struct TempFile(PathBuf);

impl TempFile {
    fn write(name: &str, bytes: &[u8]) -> Self {
        let path =
            std::env::temp_dir().join(format!("gguf-open-test-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ---------- the fixture ----------

fn fixture_bytes() -> Vec<u8> {
    let f32_data: Vec<u8> = [1.0f32, -2.5, 3.25, 0.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    // F16: 1.0 (0x3c00) and 0.5 (0x3800).
    let f16_data: Vec<u8> = [0x3c00u16, 0x3800]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    build(
        &[
            (
                "general.architecture",
                MetadataValue::String("testarch".into()),
            ),
            ("test.count", MetadataValue::UInt32(42)),
        ],
        &[
            tensor("a.weight", &[4], 0, f32_data),
            tensor("b.weight", &[2], 1, f16_data),
        ],
        32,
    )
}

// ---------- parity ----------

#[test]
fn open_matches_from_bytes_on_every_observable() {
    let bytes = fixture_bytes();
    let tmp = TempFile::write("parity.gguf", &bytes);

    let mapped = GgufFile::open(&tmp.0).unwrap();
    let owned = GgufFile::from_bytes(bytes).unwrap();

    assert_eq!(mapped.version(), owned.version());
    assert_eq!(mapped.file_len(), owned.file_len());
    assert_eq!(mapped.data_start(), owned.data_start());
    assert_eq!(mapped.alignment(), owned.alignment());
    assert_eq!(mapped.architecture().unwrap(), "testarch");
    assert_eq!(mapped.metadata().get_u32("test.count").unwrap(), 42);
    assert_eq!(
        mapped.tensor_names().collect::<Vec<_>>(),
        owned.tensor_names().collect::<Vec<_>>()
    );
    for name in owned.tensor_names() {
        assert_eq!(
            mapped.tensor_bytes(name).unwrap(),
            owned.tensor_bytes(name).unwrap(),
            "raw bytes differ for {name}"
        );
        assert_eq!(
            mapped.load_f32(name).unwrap(),
            owned.load_f32(name).unwrap(),
            "dequantized values differ for {name}"
        );
    }
    assert_eq!(
        mapped.load_f32("a.weight").unwrap(),
        vec![1.0, -2.5, 3.25, 0.0]
    );
    assert_eq!(mapped.load_f32("b.weight").unwrap(), vec![1.0, 0.5]);
    let err = mapped.tensor_bytes("nope.weight").unwrap_err();
    assert!(matches!(err, GgufError::MissingTensor(_)));
}

// ---------- named failures, identical on both paths ----------

#[test]
fn open_missing_file_is_io_not_found() {
    let path = std::env::temp_dir().join(format!(
        "gguf-open-test-{}-does-not-exist.gguf",
        std::process::id()
    ));
    let err = GgufFile::open(&path).unwrap_err();
    match err {
        GgufError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        other => panic!("expected Io(NotFound), got {other:?}"),
    }
}

#[test]
fn open_empty_file_fails_like_from_bytes() {
    let tmp = TempFile::write("empty.gguf", &[]);
    let open_err = GgufFile::open(&tmp.0).unwrap_err();
    let bytes_err = GgufFile::from_bytes(Vec::new()).unwrap_err();
    // The in-memory path names this Truncated("magic"); a zero-length
    // file cannot be mapped, and open must not let that turn into a
    // different (Io) failure.
    assert!(matches!(
        open_err,
        GgufError::Truncated {
            context: "magic",
            ..
        }
    ));
    assert_eq!(open_err.to_string(), bytes_err.to_string());
}

#[test]
fn open_bad_magic_fails_like_from_bytes() {
    let mut bytes = fixture_bytes();
    bytes[0] = b'X';
    let tmp = TempFile::write("bad-magic.gguf", &bytes);
    let open_err = GgufFile::open(&tmp.0).unwrap_err();
    let bytes_err = GgufFile::from_bytes(bytes).unwrap_err();
    assert!(matches!(open_err, GgufError::BadMagic { .. }));
    assert_eq!(open_err.to_string(), bytes_err.to_string());
}

#[test]
fn open_truncated_file_fails_like_from_bytes() {
    let full = fixture_bytes();
    // Two cut points: inside the header (directory incomplete) and
    // inside the tensor data (directory fine, data out of bounds).
    for cut in [full.len() / 3, full.len() - 2] {
        let truncated = &full[..cut];
        let tmp = TempFile::write(&format!("truncated-{cut}.gguf"), truncated);
        let open_err = GgufFile::open(&tmp.0).unwrap_err();
        let bytes_err = GgufFile::from_bytes(truncated.to_vec()).unwrap_err();
        assert_eq!(
            open_err.to_string(),
            bytes_err.to_string(),
            "error differs at cut {cut}"
        );
    }
}
