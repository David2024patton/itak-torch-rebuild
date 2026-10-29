//! Extract a GGUF file's embedded tokenizer data to a `.tokdata` file.
//!
//! The tokenizer crate deliberately does not depend on this crate, so
//! conformance work needs the vocabulary in a neutral interchange
//! format. This tool produces it: it opens a GGUF file (which loads the
//! whole file, so run it on the machine that holds the model), reads
//! `tokenizer.ggml.*` through [`GgufFile::tokenizer`] plus the unknown
//! token id from raw metadata, and writes the arrays out framed.
//!
//! Usage: `cargo run --release --example tokenizer_extract -- <model.gguf> <out.tokdata>`
//!
//! # The `.tokdata` format (version 1)
//!
//! Line-oriented ASCII framing; string payloads are length-prefixed raw
//! UTF-8 bytes so any token text survives verbatim:
//!
//! ```text
//! ITAK-TOKDATA-1
//! model <tokenizer.ggml.model, or ->
//! pre <tokenizer.ggml.pre, or ->
//! bos <id, or ->
//! eos <id, or ->
//! unk <id, or ->
//! pad <id, or ->
//! tokens <count>
//! <byte length>\n<raw bytes>\n   (repeated count times)
//! token_types <count, 0 when the file records none>
//! <one signed decimal per line>  (repeated count times)
//! merges <count>
//! <byte length>\n<raw bytes>\n   (repeated count times)
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;

use gguf::GgufFile;

fn opt_id(id: Option<u32>) -> String {
    id.map_or_else(|| "-".to_string(), |v| v.to_string())
}

fn write_framed(out: &mut impl Write, s: &str) -> std::io::Result<()> {
    writeln!(out, "{}", s.len())?;
    out.write_all(s.as_bytes())?;
    writeln!(out)
}

fn run(model_path: &str, out_path: &str) -> Result<(), String> {
    let file = GgufFile::open(Path::new(model_path)).map_err(|e| e.to_string())?;
    let info = file.tokenizer().map_err(|e| e.to_string())?;
    let md = file.metadata();
    let model = md.opt_string("tokenizer.ggml.model").map_err(|e| e.to_string())?;
    let pre = md.opt_string("tokenizer.ggml.pre").map_err(|e| e.to_string())?;
    let unk = md
        .opt_u32("tokenizer.ggml.unknown_token_id")
        .map_err(|e| e.to_string())?;

    let mut out = BufWriter::new(File::create(out_path).map_err(|e| e.to_string())?);
    let write = |out: &mut BufWriter<File>| -> std::io::Result<()> {
        writeln!(out, "ITAK-TOKDATA-1")?;
        writeln!(out, "model {}", model.unwrap_or("-"))?;
        writeln!(out, "pre {}", pre.unwrap_or("-"))?;
        writeln!(out, "bos {}", opt_id(info.bos_token_id))?;
        writeln!(out, "eos {}", opt_id(info.eos_token_id))?;
        writeln!(out, "unk {}", opt_id(unk))?;
        writeln!(out, "pad {}", opt_id(info.padding_token_id))?;
        writeln!(out, "tokens {}", info.tokens.len())?;
        for token in &info.tokens {
            write_framed(out, token)?;
        }
        match &info.token_types {
            Some(types) => {
                writeln!(out, "token_types {}", types.len())?;
                for ty in types {
                    writeln!(out, "{ty}")?;
                }
            }
            None => writeln!(out, "token_types 0")?,
        }
        let merges: &[String] = info.merges.as_deref().unwrap_or(&[]);
        writeln!(out, "merges {}", merges.len())?;
        for merge in merges {
            write_framed(out, merge)?;
        }
        Ok(())
    };
    write(&mut out).map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;

    println!(
        "extracted {} tokens, {} merges -> {out_path}",
        info.tokens.len(),
        info.merges.as_deref().unwrap_or(&[]).len()
    );
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(model), Some(out)) = (args.next(), args.next()) else {
        eprintln!("usage: tokenizer_extract <model.gguf> <out.tokdata>");
        return ExitCode::from(2);
    };
    match run(&model, &out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
