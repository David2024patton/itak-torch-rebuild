//! Dequantize a raw byte range (for example one tensor block extracted
//! from a real GGUF with `dd`) and print the f32 values, one per line.
//! Used to cross-check this crate's dequantizers against reference
//! implementations on real model bytes.
//!
//! Usage: `cargo run --example probe -- <type> <file> <elements> [count]`
//! where type is one of f32, f16, bf16, q8_0, q4_k, q5_k, q6_k.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: probe <type> <file> <elements> [count]");
        return ExitCode::from(2);
    }
    let ty = match args[1].as_str() {
        "f32" => gguf::GgmlType::F32,
        "f16" => gguf::GgmlType::F16,
        "bf16" => gguf::GgmlType::Bf16,
        "q8_0" => gguf::GgmlType::Q8_0,
        "q4_k" => gguf::GgmlType::Q4K,
        "q5_k" => gguf::GgmlType::Q5K,
        "q6_k" => gguf::GgmlType::Q6K,
        other => {
            eprintln!("unknown type {other}");
            return ExitCode::from(2);
        }
    };
    let data = match std::fs::read(&args[2]) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("read error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let elements: usize = match args[3].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("bad element count {}", args[3]);
            return ExitCode::from(2);
        }
    };
    let count: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let out = gguf::dequant::dequantize(ty, &data, elements);
    for v in out.iter().take(count) {
        println!("{v:?}");
    }
    ExitCode::SUCCESS
}
