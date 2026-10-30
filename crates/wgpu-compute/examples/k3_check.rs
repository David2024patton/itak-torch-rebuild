//! K3 correctness on the real file: run the real Q5_K output
//! head through the wgpu backend and compare against the CPU
//! reference (the `gguf` crate's dequantizer plus `torch-math`'s
//! matvec), the same cross-check style as `k1_check`. The head
//! is 202,048 rows, so this is also the two-dimensional
//! dispatch's real-file proof (101,024 workgroups at the
//! shipped two-rows-per-workgroup shape, past the 65,535
//! per-dimension cap).
//!
//! Usage:
//!   k3_check [--model PATH] [--tensor NAME] [--seed N] [--variant NAME]
//!
//! Defaults: the Glimmer Q4_K_M file on the server and its
//! output head. Run it on the machine whose GPU is under test
//! (the 7900 XTX server): the default selection policy refuses
//! machines without a discrete Vulkan GPU, and this tool
//! reports that as a SKIP (exit 0), not a failure. A real
//! mismatch exits 1. The CPU reference dequantizes the whole
//! head (about 5.4 GB of f32), so this tool wants a machine
//! with the RAM for it; the server has it.
//!
//! `--variant` selects the K3 shape under test (see
//! `K3Shape::from_name`); the default is the shipped shape for
//! the device, i.e. exactly what `Compute::lm_head` would run.

use std::path::Path;
use std::process::ExitCode;

use gguf::{GgmlType, GgufFile};
use wgpu_compute::kernels::k3_q5k_gemv::{shipped_shape, K3Pipeline, K3Shape};
use wgpu_compute::{DetectPolicy, WgpuCompute, WgpuError};

/// Deterministic pseudo-random f32s in [-0.5, 0.5): a 64-bit LCG,
/// so the check input is reproducible without a rand dependency.
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").unwrap_or_else(|| {
        "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf".to_string()
    });
    let tensor = arg(&args, "--tensor").unwrap_or_else(|| "output.weight".to_string());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let variant = arg(&args, "--variant").unwrap_or_else(|| "shipped".to_string());

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
    if info.ggml_type() != Some(GgmlType::Q5K) {
        eprintln!(
            "tensor {tensor} is {}, not Q5_K; K3 checks Q5_K tensors only",
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
        "tensor {tensor}: [{inn} x {out}] Q5_K, {} bytes",
        bytes.len()
    );

    let input = lcg_input(seed, inn);
    // CPU reference: gguf's dequantizer is ggml-exact, and matvec
    // sums in row order; the GPU sums in a tree, so the comparison
    // tolerance is relative to the output's own scale.
    let weights = gguf::dequant::dequantize(GgmlType::Q5K, &bytes, inn * out);
    let expected = torch_math::matvec(&weights, out, inn, &input).expect("cpu reference matvec");

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

    let shape = if variant == "shipped" {
        shipped_shape(ctx.device.features())
    } else {
        match K3Shape::from_name(&variant) {
            Some(s) => s,
            None => {
                eprintln!("unknown variant {variant:?} (expected r<rows>[sg] or shipped)");
                return ExitCode::from(2);
            }
        }
    };
    println!(
        "shape: {} (rows {}, workgroup {}, subgroup {})",
        shape.name(),
        shape.rows,
        shape.workgroup,
        shape.subgroup
    );
    let pipeline = match K3Pipeline::with_shape(&ctx.device, shape) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("k3 pipeline for shape {} failed: {e}", shape.name());
            return ExitCode::from(2);
        }
    };
    let weights = match pipeline.upload(ctx, &tensor, &bytes, inn, out) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("upload of {tensor} failed: {e}");
            return ExitCode::from(2);
        }
    };
    let got = match pipeline.run(ctx, &weights, &input) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gpu run failed: {e}");
            return ExitCode::from(2);
        }
    };

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
    println!(
        "outputs {out}: max_abs_err {max_abs:.6} at output scale {scale:.3} (rel {rel:.3e}), cosine {cosine:.9}"
    );
    // f32 reassociation between a sequential CPU sum and the GPU
    // tree reduction lands near 1e-6 relative; the bound leaves
    // two orders of headroom and still catches any layout error,
    // which would be off by whole dequantized values.
    if rel < 1e-4 && cosine > 0.999_999 {
        println!("K3 CHECK PASS");
        ExitCode::SUCCESS
    } else {
        println!("K3 CHECK FAIL");
        ExitCode::from(1)
    }
}
