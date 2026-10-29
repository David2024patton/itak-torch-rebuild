//! Tokenizer conformance harness (the TODO T-2.3 proof).
//!
//! Compares this crate's [`Tokenizer::encode`] token-for-token against
//! reference tokenizations produced by llama.cpp from the same model's
//! vocabulary, and checks that [`Tokenizer::decode`] of the reference
//! ids reproduces the prompt exactly.
//!
//! Usage:
//! `cargo run --release -p tokenizer --example conformance -- <vocab> <reference.tokref>`
//!
//! `<vocab>` is either a `.gguf` file (read through the `gguf` crate,
//! a dev-dependency only: the library itself stays decoupled) or a
//! `.tokdata` file produced by the gguf crate's `tokenizer_extract`
//! example, which is how a model too large to move (Glimmer is 15 GB)
//! is fed in: extract on the machine that holds the model, copy the
//! small `.tokdata` over.
//!
//! `<reference.tokref>` is produced by `conformance/make_reference.py`
//! from a live llama-server's `/tokenize` endpoint; the Glimmer
//! reference beside it in `conformance/` came from David's production
//! server. Exit code is success only when every case is identical.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::ExitCode;

use tokenizer::{TokenType, Tokenizer, TokenizerData};

/// Line-and-frame reader shared by the `.tokdata` and `.tokref`
/// formats: header lines are ASCII, string payloads are framed as a
/// decimal byte length line, the raw bytes, and one newline.
struct Frame {
    inner: BufReader<File>,
}

impl Frame {
    fn open(path: &Path) -> Result<Frame, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Frame {
            inner: BufReader::new(file),
        })
    }

    fn line(&mut self) -> Result<String, String> {
        let mut s = String::new();
        self.inner.read_line(&mut s).map_err(|e| e.to_string())?;
        Ok(s.trim_end_matches(['\n', '\r']).to_string())
    }

    fn expect(&mut self, want: &str) -> Result<(), String> {
        let got = self.line()?;
        if got == want {
            Ok(())
        } else {
            Err(format!("expected line {want:?}, found {got:?}"))
        }
    }

    /// Reads a `<key> <value>` header line for the given key.
    fn field(&mut self, key: &str) -> Result<String, String> {
        let line = self.line()?;
        line.strip_prefix(key)
            .and_then(|rest| rest.strip_prefix(' '))
            .map(str::to_string)
            .ok_or_else(|| format!("expected '{key} ...' line, found {line:?}"))
    }

    fn framed_string(&mut self) -> Result<String, String> {
        let len: usize = self
            .line()?
            .parse()
            .map_err(|_| "bad frame length".to_string())?;
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf).map_err(|e| e.to_string())?;
        let mut nl = [0u8; 1];
        self.inner.read_exact(&mut nl).map_err(|e| e.to_string())?;
        if nl[0] != b'\n' {
            return Err("frame not newline-terminated".to_string());
        }
        String::from_utf8(buf).map_err(|e| format!("frame is not UTF-8: {e}"))
    }
}

fn parse_id(value: &str) -> Option<u32> {
    if value == "-" {
        None
    } else {
        value.parse().ok()
    }
}

/// Adapts the gguf crate's parsed tokenizer info into this crate's
/// input data. This is the thin adapter the crate split calls for: it
/// lives in harness code, not in either library.
fn data_from_info(info: &gguf::TokenizerInfo, unknown: Option<u32>) -> TokenizerData {
    TokenizerData {
        tokens: info.tokens.clone(),
        token_types: info
            .token_types
            .as_ref()
            .map(|types| types.iter().map(|&t| TokenType::from_gguf(t)).collect())
            .unwrap_or_default(),
        merges: info.merges.clone().unwrap_or_default(),
        bos_token_id: info.bos_token_id,
        eos_token_id: info.eos_token_id,
        unknown_token_id: unknown,
        padding_token_id: info.padding_token_id,
    }
}

fn load_vocab(path: &Path) -> Result<(TokenizerData, String), String> {
    if path.extension().is_some_and(|e| e == "tokdata") {
        let mut frame = Frame::open(path)?;
        frame.expect("ITAK-TOKDATA-1")?;
        let model = frame.field("model")?;
        let pre = frame.field("pre")?;
        let bos = parse_id(&frame.field("bos")?);
        let eos = parse_id(&frame.field("eos")?);
        let unk = parse_id(&frame.field("unk")?);
        let pad = parse_id(&frame.field("pad")?);
        let n_tokens: usize = frame
            .field("tokens")?
            .parse()
            .map_err(|_| "bad token count".to_string())?;
        let mut tokens = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            tokens.push(frame.framed_string()?);
        }
        let n_types: usize = frame
            .field("token_types")?
            .parse()
            .map_err(|_| "bad token_types count".to_string())?;
        let mut token_types = Vec::with_capacity(n_types);
        for _ in 0..n_types {
            let raw: i32 = frame
                .line()?
                .parse()
                .map_err(|_| "bad token type".to_string())?;
            token_types.push(TokenType::from_gguf(raw));
        }
        let n_merges: usize = frame
            .field("merges")?
            .parse()
            .map_err(|_| "bad merge count".to_string())?;
        let mut merges = Vec::with_capacity(n_merges);
        for _ in 0..n_merges {
            merges.push(frame.framed_string()?);
        }
        let data = TokenizerData {
            tokens,
            token_types,
            merges,
            bos_token_id: bos,
            eos_token_id: eos,
            unknown_token_id: unk,
            padding_token_id: pad,
        };
        Ok((data, format!("tokdata model={model} pre={pre}")))
    } else {
        let file = gguf::GgufFile::open(path).map_err(|e| e.to_string())?;
        let info = file.tokenizer().map_err(|e| e.to_string())?;
        let unk = file
            .metadata()
            .opt_u32("tokenizer.ggml.unknown_token_id")
            .map_err(|e| e.to_string())?;
        let desc = format!("gguf {}", path.display());
        Ok((data_from_info(&info, unk), desc))
    }
}

fn load_reference(path: &Path) -> Result<Vec<(String, Vec<u32>)>, String> {
    let mut frame = Frame::open(path)?;
    frame.expect("ITAK-TOKREF-1")?;
    let n_cases: usize = frame
        .field("cases")?
        .parse()
        .map_err(|_| "bad case count".to_string())?;
    let mut cases = Vec::with_capacity(n_cases);
    for _ in 0..n_cases {
        let prompt = frame.framed_string()?;
        let n_ids: usize = frame
            .field("ids")?
            .parse()
            .map_err(|_| "bad ids count".to_string())?;
        let line = frame.line()?;
        let ids: Vec<u32> = line
            .split_whitespace()
            .map(|w| w.parse().map_err(|_| format!("bad token id {w:?}")))
            .collect::<Result<_, _>>()?;
        if ids.len() != n_ids {
            return Err(format!("expected {n_ids} ids, found {}", ids.len()));
        }
        cases.push((prompt, ids));
    }
    Ok(cases)
}

fn first_divergence(ours: &[u32], reference: &[u32]) -> Option<usize> {
    let shared = ours.len().min(reference.len());
    (0..shared)
        .find(|&i| ours[i] != reference[i])
        .or(if ours.len() != reference.len() {
            Some(shared)
        } else {
            None
        })
}

fn run(vocab_path: &Path, ref_path: &Path) -> Result<bool, String> {
    let (data, desc) = load_vocab(vocab_path)?;
    let tok = Tokenizer::new(data).map_err(|e| e.to_string())?;
    println!(
        "vocab: {desc}; {} tokens, bos={:?} eos={:?} unk={:?} pad={:?}",
        tok.vocab_size(),
        tok.bos_id(),
        tok.eos_id(),
        tok.unk_id(),
        tok.pad_id()
    );
    let cases = load_reference(ref_path)?;
    let mut exact = 0usize;
    let mut roundtrip_ok = 0usize;
    for (index, (prompt, reference)) in cases.iter().enumerate() {
        let ours = tok.encode(prompt).map_err(|e| e.to_string())?;
        let decoded = tok.decode(reference).map_err(|e| e.to_string())?;
        let ids_match = ours == *reference;
        let text_match = decoded == *prompt;
        exact += usize::from(ids_match);
        roundtrip_ok += usize::from(text_match);
        println!(
            "case {index}: {} ({} tokens) decode-roundtrip {} \u{2014} prompt {prompt:?}",
            if ids_match { "EXACT" } else { "DIFF" },
            reference.len(),
            if text_match { "ok" } else { "DIFF" },
        );
        if !ids_match {
            let at = first_divergence(&ours, reference).unwrap_or(0);
            println!("  first divergence at token index {at}");
            if let Some(&id) = reference.get(at) {
                println!("  reference[{at}] = {id} {:?}", tok.token_text(id));
            }
            if let Some(&id) = ours.get(at) {
                println!("  ours     [{at}] = {id} {:?}", tok.token_text(id));
            }
            println!("  reference ids: {reference:?}");
            println!("  ours ids:      {ours:?}");
        }
        if !text_match {
            println!("  decoded reference ids: {decoded:?}");
        }
    }
    println!(
        "conformance: {exact}/{} prompts token-identical; {roundtrip_ok}/{} decode round-trips exact",
        cases.len(),
        cases.len()
    );
    Ok(exact == cases.len() && roundtrip_ok == cases.len())
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let (Some(vocab), Some(reference)) = (args.next(), args.next()) else {
        eprintln!("usage: conformance <vocab.gguf|vocab.tokdata> <reference.tokref>");
        return ExitCode::from(2);
    };
    match run(Path::new(&vocab), Path::new(&reference)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
