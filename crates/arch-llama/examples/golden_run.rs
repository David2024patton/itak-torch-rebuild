//! Golden-fixture candidate runner (integration run A, TODO T-2.9).
//!
//! Loads a GGUF model through this crate, tokenizes the golden
//! prompt set with the rebuild's own `tokenizer` crate fed from the
//! model's embedded GGUF tokenizer metadata, generates greedily
//! (argmax, the temperature-0 trajectory the fixtures record), and
//! writes candidate records in the fixture shape that
//! `golden/compare.py` diffs.
//!
//! The run is deliberately the whole honest path, nothing propped
//! up: our tokenizer's ids are what the model consumes (the fixture
//! comparison reports prompt-tokenization divergence separately),
//! and our forward produces every completion token.
//!
//! Two llama.cpp behaviors are mirrored explicitly, from evidence:
//!
//! - BOS: `llama-tokenize` (which produced the fixture prompt ids)
//!   prepends the BOS id per `tokenizer.ggml.add_bos_token`; the
//!   tokenizer crate never adds specials on its own, so the runner
//!   does. When the file omits the key (the TinyLlama file does),
//!   llama.cpp falls back to the tokenizer family's default, which
//!   is true for the `llama` tokenizer and false for `gpt2`; the
//!   runner mirrors that default rather than guessing per model.
//! - Stop set: generation stops on the file's declared EOS, and on
//!   the `<|endoftext|>` token when the vocabulary carries one.
//!   The Qwen2.5 file declares `eos_token_id = 151645`
//!   (<|im_end|>), yet its golden records stop on 151643
//!   (<|endoftext|>) with stop_type "eos": llama.cpp's
//!   end-of-generation set evidently covers both roles for this
//!   vocabulary, so the runner's stop set does too.
//!
//! Usage:
//!   golden_run --model <file.gguf> --prompts <prompts.jsonl>
//!              --out <candidate.jsonl> [--max-tokens 64]
//!
//! The prompts file may be `golden/prompts.jsonl` or a golden
//! fixture; both carry `id`, `category` and `prompt` per line.
//! Records are written and flushed one prompt at a time, so a
//! time-boxed run still leaves a diffable partial candidate.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use arch_llama::{LlamaConfig, LlamaModel, Weights};
use gguf::{GgufFile, MetadataValue};
use tokenizer::{TokenType, Tokenizer, TokenizerData};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("golden_run: {msg}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    model: String,
    prompts: String,
    out: String,
    max_tokens: usize,
}

fn next_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next()
        .ok_or_else(|| format!("flag {flag} needs a value"))
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut prompts = None;
    let mut out = None;
    let mut max_tokens = 64usize;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(next_value(&mut it, &arg)?),
            "--prompts" => prompts = Some(next_value(&mut it, &arg)?),
            "--out" => out = Some(next_value(&mut it, &arg)?),
            "--max-tokens" => {
                max_tokens = next_value(&mut it, &arg)?
                    .parse()
                    .map_err(|_| "bad --max-tokens value".to_string())?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        model: model.ok_or("missing --model")?,
        prompts: prompts.ok_or("missing --prompts")?,
        out: out.ok_or("missing --out")?,
        max_tokens,
    })
}

/// One prompt-set entry.
struct Prompt {
    id: String,
    category: String,
    text: String,
}

/// Extracts a string field from one flat JSON object line. Only
/// correct for the prompt-set / fixture shapes (top-level string
/// fields, no nesting), which is all this runner consumes.
fn json_string_field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let at = line.find(&pat)? + pat.len();
    let rest = line[at..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    parse_json_string(rest)
}

/// Parses a JSON string literal at the start of `s`, applying the
/// standard escapes including `\uXXXX` surrogate pairs.
fn parse_json_string(s: &str) -> Option<String> {
    let mut chars = s.chars();
    if chars.next()? != '"' {
        return None;
    }
    let mut out = String::new();
    loop {
        match chars.next()? {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'u' => {
                    let hi = hex4(&mut chars)?;
                    if (0xd800..0xdc00).contains(&hi) {
                        if chars.next()? != '\\' || chars.next()? != 'u' {
                            return None;
                        }
                        let lo = hex4(&mut chars)?;
                        let c = 0x1_0000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                        out.push(char::from_u32(c).unwrap_or('\u{fffd}'));
                    } else {
                        out.push(char::from_u32(hi).unwrap_or('\u{fffd}'));
                    }
                }
                _ => return None,
            },
            c => out.push(c),
        }
    }
}

fn hex4(chars: &mut std::str::Chars) -> Option<u32> {
    let mut v = 0u32;
    for _ in 0..4 {
        v = v * 16 + chars.next()?.to_digit(16)?;
    }
    Some(v)
}

fn load_prompts(path: &Path) -> Result<Vec<Prompt>, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut prompts = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Fixture metadata line: an object with "meta" and no "id".
        if line.contains("\"meta\"") && json_string_field(line, "id").is_none() {
            continue;
        }
        let id = json_string_field(line, "id")
            .ok_or_else(|| format!("prompt line lacks an id: {line:.60}"))?;
        let category = json_string_field(line, "category").unwrap_or_default();
        let text = json_string_field(line, "prompt")
            .ok_or_else(|| format!("prompt {id} lacks prompt text"))?;
        prompts.push(Prompt { id, category, text });
    }
    if prompts.is_empty() {
        return Err("prompt set is empty".to_string());
    }
    Ok(prompts)
}

/// Escapes a string for JSON output in the fixtures' style: quotes,
/// backslashes and control characters escaped, everything else raw
/// UTF-8 (matching python's `ensure_ascii=False`).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn ids_json(ids: &[u32]) -> String {
    let parts: Vec<String> = ids.iter().map(u32::to_string).collect();
    format!("[{}]", parts.join(", "))
}

/// Greedy choice: the first index holding the maximum logit, the
/// tie behavior of llama.cpp's greedy sampler.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let prompts = load_prompts(Path::new(&args.prompts))?;

    let t0 = Instant::now();
    let file =
        GgufFile::open(Path::new(&args.model)).map_err(|e| format!("open {}: {e}", args.model))?;
    let md = file.metadata();
    let tok_model = md
        .opt_string("tokenizer.ggml.model")
        .map_err(|e| e.to_string())?
        .unwrap_or("<absent>")
        .to_string();
    let tok_pre = md
        .opt_string("tokenizer.ggml.pre")
        .map_err(|e| e.to_string())?
        .unwrap_or("<absent>")
        .to_string();
    let add_bos = match md.get("tokenizer.ggml.add_bos_token") {
        Some(MetadataValue::Bool(b)) => *b,
        // Key absent: llama.cpp's family default (llama tokenizers
        // add BOS, gpt2 tokenizers do not). The TinyLlama file omits
        // the key and its golden prompt ids do start with BOS.
        _ => tok_model == "llama",
    };
    let info = file.tokenizer().map_err(|e| e.to_string())?;
    let unknown = md
        .opt_u32("tokenizer.ggml.unknown_token_id")
        .map_err(|e| e.to_string())?;
    let tok_data = TokenizerData {
        // The family comes from the file's own model string, never
        // from the model file's name: "llama" selects SentencePiece,
        // "gpt2" byte-level BPE (see the tokenizer crate).
        model: md
            .opt_string("tokenizer.ggml.model")
            .map_err(|e| e.to_string())?
            .map(str::to_string),
        tokens: info.tokens.clone(),
        token_types: info
            .token_types
            .as_ref()
            .map(|types| types.iter().map(|&t| TokenType::from_gguf(t)).collect())
            .unwrap_or_default(),
        merges: info.merges.clone().unwrap_or_default(),
        scores: info.scores.clone(),
        bos_token_id: info.bos_token_id,
        eos_token_id: info.eos_token_id,
        unknown_token_id: unknown,
        padding_token_id: info.padding_token_id,
    };
    let tokenizer = Tokenizer::new(tok_data).map_err(|e| format!("tokenizer: {e}"))?;

    // Stop set: the declared EOS, plus the <|endoftext|> token when
    // the vocabulary carries one (see the module docs for why).
    let mut stop: Vec<u32> = Vec::new();
    if let Some(eos) = info.eos_token_id {
        stop.push(eos);
    }
    if let Some(id) = tokenizer.token_id("<|endoftext|>") {
        if !stop.contains(&id) {
            stop.push(id);
        }
    }
    eprintln!(
        "tokenizer: model={tok_model} pre={tok_pre} vocab={} bos={:?} eos={:?} add_bos={add_bos} stop={stop:?}",
        tokenizer.vocab_size(),
        info.bos_token_id,
        info.eos_token_id,
    );

    let cfg = LlamaConfig::from_file(&file).map_err(|e| format!("config: {e}"))?;
    let weights = Weights::load(&file, &cfg).map_err(|e| format!("weights: {e}"))?;
    let model = LlamaModel::new(cfg, weights);
    eprintln!(
        "model loaded in {:.1?}: arch={} layers={}",
        t0.elapsed(),
        model.config().architecture,
        model.config().block_count,
    );

    let out_file = File::create(&args.out).map_err(|e| format!("{}: {e}", args.out))?;
    let mut out = BufWriter::new(out_file);
    let model_name = Path::new(&args.model)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    writeln!(
        out,
        "{{\"meta\": {{\"kind\": \"itak-torch-candidate\", \"engine\": \"arch-llama (Rust CPU reference forward + rebuild tokenizer)\", \"model_file\": \"{}\", \"decoding\": \"greedy argmax, temperature 0\", \"max_tokens\": {}}}}}",
        json_escape(&model_name),
        args.max_tokens,
    )
    .map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;

    for prompt in &prompts {
        let started = Instant::now();
        let mut prompt_ids = tokenizer
            .encode(&prompt.text)
            .map_err(|e| format!("encode {}: {e}", prompt.id))?;
        if add_bos {
            if let Some(bos) = info.bos_token_id {
                prompt_ids.insert(0, bos);
            }
        }
        let mut cache = model.new_cache().map_err(|e| e.to_string())?;
        let mut logits = model
            .forward(&prompt_ids, &mut cache)
            .map_err(|e| format!("forward {}: {e}", prompt.id))?;
        let mut completion: Vec<u32> = Vec::new();
        let mut ended_with_eos = false;
        for _ in 0..args.max_tokens {
            let next = argmax(&logits);
            completion.push(next);
            if stop.contains(&next) {
                ended_with_eos = true;
                break;
            }
            logits = model
                .step(next, &mut cache)
                .map_err(|e| format!("step {}: {e}", prompt.id))?;
        }
        // The terminal stop token contributes no text in the golden
        // records (its piece is empty there), so it is excluded from
        // the decoded text but kept in the id list, as in the
        // fixtures.
        let text_ids = if ended_with_eos {
            &completion[..completion.len() - 1]
        } else {
            &completion[..]
        };
        let text = tokenizer
            .decode_lossy(text_ids)
            .map_err(|e| format!("decode {}: {e}", prompt.id))?;
        writeln!(
            out,
            "{{\"id\": \"{}\", \"category\": \"{}\", \"prompt\": \"{}\", \"prompt_token_ids\": {}, \"completion_token_ids\": {}, \"completion_text\": \"{}\", \"ended_with_eos\": {}}}",
            json_escape(&prompt.id),
            json_escape(&prompt.category),
            json_escape(&prompt.text),
            ids_json(&prompt_ids),
            ids_json(&completion),
            json_escape(&text),
            ended_with_eos,
        )
        .map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
        eprintln!(
            "{}: prompt {} tokens, completion {} tokens, eos={ended_with_eos}, {:.1?}",
            prompt.id,
            prompt_ids.len(),
            completion.len(),
            started.elapsed(),
        );
    }
    eprintln!("done in {:.1?}", t0.elapsed());
    Ok(())
}
