//! Evidence tool for PLAN T-2.1 (memory-mapped GGUF backing): shows
//! that `GgufFile::open` costs no anonymous RSS proportional to file
//! size, while the pre-T-2.1 path (`std::fs::read` +
//! `GgufFile::from_bytes`) costs the full file size in anonymous RSS.
//! The OOM killer's figure is anon-rss, and the phase B footprint
//! audit (bench/phase-b-footprint-2026-11-05.md) is what this fix
//! answers, so the table splits RSS into anon and file parts.
//!
//! Usage: `cargo run -p gguf --release --example mmap_rss -- [size_mib] [dir]`
//! (default 512, system temp dir). Writes a synthetic GGUF with one
//! F32 tensor of `size_mib` MiB into `dir`, measures this process at
//! each phase, and removes the file at the end. Pass a `dir` on a
//! volume that can hold the file; a small tmpfs will not. Linux only
//! (reads /proc/self/status).

use std::io::Write;
use std::path::PathBuf;

use gguf::GgufFile;

/// (VmRSS, RssAnon, RssFile) in KiB from /proc/self/status.
fn rss_kib() -> (u64, u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status")
        .expect("mmap_rss reads /proc/self/status (Linux)");
    let field = |name: &str| -> u64 {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.trim().strip_suffix("kB"))
            .and_then(|num| num.trim().parse().ok())
            .unwrap_or(0)
    };
    (field("VmRSS:"), field("RssAnon:"), field("RssFile:"))
}

fn report(phase: &str) {
    let (rss, anon, file) = rss_kib();
    println!(
        "{phase:<28} VmRSS {:>8.1} MiB   RssAnon {:>8.1} MiB   RssFile {:>8.1} MiB",
        rss as f64 / 1024.0,
        anon as f64 / 1024.0,
        file as f64 / 1024.0,
    );
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

/// Write a GGUF v3 file holding one F32 tensor of `data_bytes` zero
/// bytes (element count `data_bytes / 4`), streaming the data section
/// in chunks so the writer itself never holds the file in memory.
fn write_big_gguf(path: &PathBuf, data_bytes: u64) {
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    header.extend_from_slice(&1u64.to_le_bytes()); // metadata kv count
                                                   // general.architecture = "rss-probe"
    header.extend_from_slice(&20u64.to_le_bytes());
    header.extend_from_slice(b"general.architecture");
    header.extend_from_slice(&8u32.to_le_bytes()); // value type: string
    header.extend_from_slice(&9u64.to_le_bytes());
    header.extend_from_slice(b"rss-probe");
    // Tensor directory: "big.weight", 1 dim, F32 (type 0), offset 0.
    header.extend_from_slice(&10u64.to_le_bytes());
    header.extend_from_slice(b"big.weight");
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&(data_bytes / 4).to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    let data_start = align_up(header.len() as u64, 32);
    header.resize(data_start as usize, 0);

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&header).unwrap();
    let chunk = vec![0u8; 8 * 1024 * 1024];
    let mut remaining = data_bytes;
    while remaining > 0 {
        let n = remaining.min(chunk.len() as u64) as usize;
        file.write_all(&chunk[..n]).unwrap();
        remaining -= n as u64;
    }
    file.sync_all().unwrap();
}

fn main() {
    let size_mib: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(512);
    let data_bytes = size_mib * 1024 * 1024;
    let dir = std::env::args()
        .nth(2)
        .map_or_else(std::env::temp_dir, PathBuf::from);
    let path = dir.join(format!("gguf-mmap-rss-{}.gguf", std::process::id()));
    write_big_gguf(&path, data_bytes);
    println!(
        "synthetic GGUF: {} (tensor data {size_mib} MiB)",
        path.display()
    );

    report("start");
    {
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(file.architecture().unwrap(), "rss-probe");
        report("mapped open");
        let tensor = file.tensor_bytes("big.weight").unwrap();
        assert_eq!(tensor.len() as u64, data_bytes);
        // Touch three pages: first, middle, last.
        let probe =
            tensor[0] as u64 + tensor[tensor.len() / 2] as u64 + tensor[tensor.len() - 1] as u64;
        std::hint::black_box(probe);
        report("mapped touch 3 pages");
        // Touch every page of the tensor data.
        let mut sum: u64 = 0;
        for page in tensor.chunks(4096) {
            sum += page[0] as u64;
        }
        std::hint::black_box(sum);
        report("mapped touch every page");
    }
    report("mapped dropped");
    {
        let bytes = std::fs::read(&path).unwrap();
        let file = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(file.architecture().unwrap(), "rss-probe");
        report("owned read + from_bytes");
    }
    let _ = std::fs::remove_file(&path);
}
