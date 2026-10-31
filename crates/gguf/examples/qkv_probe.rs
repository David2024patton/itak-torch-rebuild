//! QKV source probe (layer-diff follow-up to integration run A).
//!
//! llama.cpp prefers a fused `attn_qkv.weight` when a GGUF carries
//! one and slices it as contiguous [Q; K; V] row blocks; this crate
//! loads the split `attn_q/k/v.weight` tensors. When a file carries
//! both, the two sources must agree or one of them is not what it
//! claims to be. This probe dequantizes both sources for one layer,
//! applies each to a caller-supplied attn_norm dump (the Rust
//! layer_dump output, proven to match the reference to 1e-6), and
//! writes every candidate Q/K as .bin files for numerical
//! comparison against the reference's Qcur/Kcur dump.
//!
//! Usage:
//!   qkv_probe --model <file.gguf> --attn-norm <L00_attn_norm.bin>
//!             --out <dir> [--layer 0]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gguf::GgufFile;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("qkv_probe: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn next_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next()
        .ok_or_else(|| format!("flag {flag} needs a value"))
}

fn write_bin(dir: &Path, name: &str, data: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let path = dir.join(format!("{name}.bin"));
    fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// out = h * W^T + bias, W in [out][in] row-major (GGUF order).
fn proj(h: &[f32], rows: usize, w: &[f32], out_w: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let in_w = w.len() / out_w;
    let mut out = vec![0.0f32; rows * out_w];
    for r in 0..rows {
        let hr = &h[r * in_w..(r + 1) * in_w];
        for o in 0..out_w {
            let wr = &w[o * in_w..(o + 1) * in_w];
            let mut acc: f32 = hr.iter().zip(wr.iter()).map(|(a, b)| a * b).sum();
            if let Some(b) = bias {
                acc += b[o];
            }
            out[r * out_w + o] = acc;
        }
    }
    out
}

fn run() -> Result<(), String> {
    let mut model = None;
    let mut attn_norm_path = None;
    let mut out_dir = None;
    let mut layer = 0usize;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(next_value(&mut it, &arg)?),
            "--attn-norm" => attn_norm_path = Some(next_value(&mut it, &arg)?),
            "--out" => out_dir = Some(next_value(&mut it, &arg)?),
            "--layer" => {
                layer = next_value(&mut it, &arg)?
                    .parse()
                    .map_err(|_| "bad --layer".to_string())?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let model = model.ok_or("missing --model")?;
    let attn_norm_path = attn_norm_path.ok_or("missing --attn-norm")?;
    let out_dir = PathBuf::from(out_dir.ok_or("missing --out")?);
    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    let file = GgufFile::open(Path::new(&model)).map_err(|e| format!("open: {e}"))?;
    let p = format!("blk.{layer}");
    for suffix in [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_qkv.weight",
        "attn_q.bias",
        "attn_k.bias",
        "attn_v.bias",
        "attn_qkv.bias",
    ] {
        let name = format!("{p}.{suffix}");
        match file.tensor_info(&name) {
            Ok(info) => eprintln!(
                "{name}: dims={:?} type={} elements={}",
                info.dims(),
                info.type_name(),
                info.element_count()
            ),
            Err(_) => eprintln!("{name}: ABSENT"),
        }
    }

    let raw = fs::read(&attn_norm_path).map_err(|e| format!("{attn_norm_path}: {e}"))?;
    let h: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let hidden = 1536usize;
    let rows = h.len() / hidden;

    let load = |suffix: &str| -> Result<Option<Vec<f32>>, String> {
        let name = format!("{p}.{suffix}");
        if file.has_tensor(&name) {
            Ok(Some(
                file.load_f32(&name).map_err(|e| format!("{name}: {e}"))?,
            ))
        } else {
            Ok(None)
        }
    };
    let wq = load("attn_q.weight")?.ok_or("attn_q.weight absent")?;
    let wk = load("attn_k.weight")?.ok_or("attn_k.weight absent")?;
    let bq = load("attn_q.bias")?;
    let bk = load("attn_k.bias")?;
    let q_w = 1536usize;
    let k_w = 256usize;

    write_bin(
        &out_dir,
        "q_split",
        &proj(&h, rows, &wq, q_w, bq.as_deref()),
    )?;
    write_bin(
        &out_dir,
        "k_split",
        &proj(&h, rows, &wk, k_w, bk.as_deref()),
    )?;

    if let Some(wqkv) = load("attn_qkv.weight")? {
        let total_w = q_w + 2 * k_w;
        let bqkv = load("attn_qkv.bias")?;
        let bias_q: Option<&[f32]> = match (&bqkv, &bq) {
            (Some(b), _) => Some(&b[..q_w]),
            (None, Some(b)) => Some(b.as_slice()),
            (None, None) => None,
        };
        let bias_k: Option<&[f32]> = match (&bqkv, &bk) {
            (Some(b), _) => Some(&b[q_w..q_w + k_w]),
            (None, Some(b)) => Some(b.as_slice()),
            (None, None) => None,
        };
        // Contiguous [Q; K; V] row blocks: the llama.cpp slicing.
        let wq_f = &wqkv[..q_w * hidden];
        let wk_f = &wqkv[q_w * hidden..(q_w + k_w) * hidden];
        write_bin(&out_dir, "q_fused", &proj(&h, rows, wq_f, q_w, bias_q))?;
        write_bin(&out_dir, "k_fused", &proj(&h, rows, wk_f, k_w, bias_k))?;
        // Interleaved GQA slicing: per KV group, [6 Q heads; 1 K; 1 V].
        let group_q = q_w / 2; // 768 rows of Q per KV group
        let stride = group_q + 2 * k_w;
        let mut wq_i = vec![0.0f32; q_w * hidden];
        let mut wk_i = vec![0.0f32; k_w * hidden];
        for g in 0..2usize {
            let src = g * stride * hidden;
            wq_i[g * group_q * hidden..(g + 1) * group_q * hidden]
                .copy_from_slice(&wqkv[src..src + group_q * hidden]);
            let ks = src + group_q * hidden;
            wk_i[g * k_w * hidden..(g + 1) * k_w * hidden]
                .copy_from_slice(&wqkv[ks..ks + k_w * hidden]);
        }
        write_bin(
            &out_dir,
            "q_fused_interleaved",
            &proj(&h, rows, &wq_i, q_w, bias_q),
        )?;
        write_bin(
            &out_dir,
            "k_fused_interleaved",
            &proj(&h, rows, &wk_i, k_w, bias_k),
        )?;
        let _ = total_w;
    }
    eprintln!("qkv_probe: wrote candidates to {}", out_dir.display());
    Ok(())
}
