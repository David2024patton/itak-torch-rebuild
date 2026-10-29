//! Dump a GGUF file's header, model config, and tensor inventory.
//!
//! Usage: `cargo run --example dump -- <path.gguf>`
//! Set `GGUF_DEBUG=1` to see the loader's structured diagnostics as it parses.

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: dump <path.gguf>");
        return ExitCode::from(2);
    };
    let file = match gguf::GgufFile::open(Path::new(&path)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("file: {path}");
    println!("version: {}", file.version());
    println!("file_len: {} bytes", file.file_len());
    println!("alignment: {}", file.alignment());
    println!("data_start: {}", file.data_start());
    println!("metadata entries: {}", file.metadata().len());
    match file.architecture() {
        Ok(arch) => println!("architecture: {arch}"),
        Err(e) => println!("architecture: <error: {e}>"),
    }
    match file.model_config() {
        Ok(cfg) => println!("model_config: {cfg:?}"),
        Err(e) => println!("model_config: <error: {e}>"),
    }
    match file.tokenizer() {
        Ok(tok) => println!(
            "tokenizer: {} tokens, bos={:?} eos={:?}",
            tok.tokens.len(),
            tok.bos_token_id,
            tok.eos_token_id
        ),
        Err(e) => println!("tokenizer: <error: {e}>"),
    }
    println!("tensors: {}", file.tensors().len());
    for t in file.tensors() {
        println!(
            "  {} type={} dims={:?} elements={} bytes={:?}",
            t.name(),
            t.type_name(),
            t.dims(),
            t.element_count(),
            t.byte_len()
        );
    }
    ExitCode::SUCCESS
}
