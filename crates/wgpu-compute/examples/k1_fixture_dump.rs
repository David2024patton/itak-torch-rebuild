//! K1 fixture dump for cross-language kernel benches (bench
//! tooling for the Mojo kernel test, bench/mojo-kernel-test-
//! 2026-11-07.md; not part of the engine and not a shipped path).
//!
//! Writes the fixture files a non-Rust K1 port needs to run the
//! SAME math on the SAME data as `k1_check`:
//!   weights_q4k.bin  raw Q4_K bytes of the tensor, file order
//!   input_f32.bin    the check's deterministic LCG input (f32 LE)
//!   ref_f32.bin      CPU reference output (gguf dequantizer plus
//!                    torch-math matvec, the harness's own path)
//!   gpu_f32.bin      the shipped wgpu K1 pipeline's output
//!   fixture.txt      shapes and the gpu-vs-ref cosine/rel line in
//!                    exactly the format `k1_check` prints
//!
//! Usage:
//!   k1_fixture_dump --out DIR [--model PATH] [--tensor NAME]
//!                   [--seed N]
//!
//! Defaults match `k1_check`: the Glimmer Q4_K_M file, its first
//! FFN gate tensor, seed 42, shipped K1 shape. On a machine with
//! no suitable GPU the tool reports SKIP (exit 0) after writing
//! the CPU-side files, like `k1_check`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gguf::{GgmlType, GgufFile};
use wgpu_compute::kernels::k1_q4k_gemv::{shipped_shape, K1Pipeline};
use wgpu_compute::{DetectPolicy, WgpuCompute, WgpuError};

/// Deterministic pseudo-random f32s in [-0.5, 0.5): the same
/// 64-bit LCG `k1_check` uses, so the input is bit-identical.
fn lcg_input(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 40) as u32;
            bits as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
    let path = dir.join(name);
    std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
    Ok(path)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(out_dir) = arg(&args, "--out") else {
        eprintln!("usage: k1_fixture_dump --out DIR [--model PATH] [--tensor NAME] [--seed N]");
        return ExitCode::from(2);
    };
    let out_dir = PathBuf::from(out_dir);
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("cannot create {}: {e}", out_dir.display());
        return ExitCode::from(2);
    }
    let model = arg(&args, "--model").unwrap_or_else(|| {
        "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf".to_string()
    });
    let tensor = arg(&args, "--tensor").unwrap_or_else(|| "blk.0.ffn_gate.weight".to_string());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let file = match GgufFile::open(Path::new(&model)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {model}: {e}");
            return ExitCode::from(2);
        }
    };
    let info = match file.tensor_info(&tensor) {
        Ok(i) => i.clone(),
        Err(e) => {
            eprintln!("tensor {tensor} not readable in {model}: {e}");
            return ExitCode::from(2);
        }
    };
    if info.ggml_type() != Some(GgmlType::Q4K) {
        eprintln!(
            "REFUSAL k1_fixture_dump: tensor {tensor} is {}, not Q4_K",
            info.type_name()
        );
        return ExitCode::from(2);
    }
    let dims = info.dims().to_vec();
    let (inn, out) = (dims[0] as usize, dims[1] as usize);
    let bytes = match file.tensor_bytes(&tensor) {
        Ok(b) => b.to_vec(),
        Err(e) => {
            eprintln!("cannot read bytes of {tensor}: {e}");
            return ExitCode::from(2);
        }
    };
    println!(
        "tensor {tensor}: [{inn} x {out}] Q4_K, {} bytes",
        bytes.len()
    );

    let input = lcg_input(seed, inn);
    let weights = gguf::dequant::dequantize(GgmlType::Q4K, &bytes, inn * out);
    let expected = torch_math::matvec(&weights, out, inn, &input).expect("cpu reference matvec");

    for (name, data) in [
        ("weights_q4k.bin", bytes.as_slice()),
        ("input_f32.bin", &f32_bytes(&input)),
        ("ref_f32.bin", &f32_bytes(&expected)),
    ] {
        if let Err(e) = write_file(&out_dir, name, data) {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    }

    let backend = match WgpuCompute::with_policy(&DetectPolicy::default()) {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            println!(
                "SKIP: no suitable GPU on this machine (enumerated: [{}])",
                seen.join("; ")
            );
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("backend construction failed: {e}");
            return ExitCode::from(2);
        }
    };
    let ctx = backend.context();
    println!("adapter: {}", ctx.adapter.name);
    let shape = shipped_shape(ctx.device.features());
    println!(
        "shape: {} (rows {}, workgroup {}, subgroup {}, f16 activations {}, legacy {})",
        shape.name(),
        shape.rows,
        shape.workgroup,
        shape.subgroup,
        shape.f16_activations,
        shape.legacy
    );
    let pipeline = match K1Pipeline::with_shape(&ctx.device, shape) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("k1 pipeline for shape {} failed: {e}", shape.name());
            return ExitCode::from(2);
        }
    };
    let uploaded = match pipeline.upload(ctx, &tensor, &bytes, inn, out) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("upload of {tensor} failed: {e}");
            return ExitCode::from(2);
        }
    };
    let got = match pipeline.run(ctx, &uploaded, &input) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gpu run failed: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = write_file(&out_dir, "gpu_f32.bin", &f32_bytes(&got)) {
        eprintln!("{e}");
        return ExitCode::from(2);
    }

    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    let mut dot = 0.0f64;
    let mut norm_g = 0.0f64;
    let mut norm_e = 0.0f64;
    for (g, e) in got.iter().zip(expected.iter()) {
        max_abs = max_abs.max((g - e).abs());
        scale = scale.max(e.abs());
        dot += f64::from(*g) * f64::from(*e);
        norm_g += f64::from(*g) * f64::from(*g);
        norm_e += f64::from(*e) * f64::from(*e);
    }
    let cosine = dot / (norm_g.sqrt() * norm_e.sqrt());
    let rel = max_abs / scale.max(1.0);
    let summary = format!(
        "tensor {tensor}: [{inn} x {out}] Q4_K, {} bytes\n\
         outputs {out}: max_abs_err {max_abs:.6} at output scale {scale:.3} (rel {rel:.3e}), cosine {cosine:.9}\n",
        bytes.len()
    );
    print!("{summary}");
    if let Err(e) = write_file(&out_dir, "fixture.txt", summary.as_bytes()) {
        eprintln!("{e}");
        return ExitCode::from(2);
    }
    if rel < 1e-4 && cosine > 0.999_999 {
        println!("K1 FIXTURE PASS");
        ExitCode::SUCCESS
    } else {
        println!("K1 FIXTURE FAIL");
        ExitCode::from(1)
    }
}
