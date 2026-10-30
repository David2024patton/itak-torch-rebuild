//! Integration run B: Glimmer's first generation on the new Rust
//! stack (TODO: integration stage), driven end to end through the
//! real crates: `gguf` parses the file, `itak-loader` loads config
//! and weights behind the coverage gate, the `tokenizer` crate
//! tokenizes, and `muse-glimmer`'s forward runs on `cpu-compute`.
//!
//! Usage: `glimmer_generate <path-to-gguf> <prompt> [n_tokens] [--ids-only]`
//!
//! The prompt ids are printed before any weight work happens, so the
//! identical id sequence can be handed to a reference engine (the
//! llama.cpp comparison this run exists for). `--ids-only` stops
//! there: tokenize and exit, no weight load, no forward.
//!
//! Output is line-oriented and flushed per step, so a long CPU run
//! shows its progress token by token: phase timings, the prompt ids,
//! and per generated position the top-5 candidates (id, raw token
//! text, logit, softmax probability over the full vocabulary) plus
//! the greedy pick and its wall time.
//!
//! This is a correctness runner, not a benchmark: `cpu-compute`
//! dequantizes weights on every projection, so a 30B token is
//! minutes-class. Timings are reported honestly, per token.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use muse_glimmer::{GlimmerModel, PositionCache, StderrDiag};
use tokenizer::{TokenType, Tokenizer, TokenizerData};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(path), Some(prompt)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: glimmer_generate <path-to-gguf> <prompt> [n_tokens] [--ids-only]");
        return ExitCode::from(2);
    };
    let ids_only = args.iter().any(|a| a == "--ids-only");
    let n_tokens: usize = args
        .get(3)
        .filter(|a| !a.starts_with("--"))
        .map_or(8, |a| a.parse().unwrap_or(8));
    match run(path, prompt, n_tokens, ids_only) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("FAIL: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Adapts the gguf crate's parsed tokenizer info into the tokenizer
/// crate's input data. Same thin adapter as the conformance harness
/// (TODO T-2.3); it lives in harness code, not in either library.
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

/// The `k` highest logits as (id, logit, softmax probability over
/// the whole vocabulary), best first. Probabilities are computed in
/// f64 from the forward's finished (softcapped, scaled) logits.
///
/// Ordering is fully deterministic: descending logit, ties broken
/// by ascending token id. The greedy pick (rank 0) is therefore the
/// FIRST maximum, the same choice a plain first-max scan makes;
/// with an unspecified tie order, two runs could in principle
/// continue a sequence with different tokens on an exact tie.
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32, f64)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f64 = logits
        .iter()
        .map(|&z| f64::from(z - max).exp())
        .sum::<f64>();
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    idx.iter()
        .map(|&id| {
            let z = logits[id as usize];
            (id, z, f64::from(z - max).exp() / denom)
        })
        .collect()
}

fn print_top(label: &str, top: &[(u32, f32, f64)], vocab: &[String]) {
    for (rank, (id, logit, prob)) in top.iter().enumerate() {
        println!(
            "{label} rank={rank} id={id} tok={:?} logit={logit:.6} prob={prob:.6}",
            vocab[*id as usize]
        );
    }
    std::io::stdout().flush().ok();
}

fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

fn run(path: &str, prompt: &str, n_tokens: usize, ids_only: bool) -> Result<(), String> {
    let t_open = Instant::now();
    let file = gguf::GgufFile::open(Path::new(path)).map_err(|e| e.to_string())?;
    println!("phase=open_file ms={}", t_open.elapsed().as_millis());
    println!("file: {path} bytes={}", file.file_len());

    let cfg = itak_loader::load_config(&file).map_err(|e| e.to_string())?;
    println!(
        "config: layers={} hidden={} vocab={} bos={:?}",
        cfg.block_count,
        cfg.hidden,
        cfg.vocab_size,
        file.tokenizer().map(|t| t.bos_token_id).unwrap_or(None)
    );

    // Tokenize before any weight work: the printed id sequence is
    // the exact input both engines are compared on.
    let info = file.tokenizer().map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::new(data_from_info(&info, None)).map_err(|e| e.to_string())?;
    let body = tokenizer.encode(prompt).map_err(|e| e.to_string())?;
    let add_bos = file
        .metadata()
        .get_bool("tokenizer.ggml.add_bos_token")
        .unwrap_or(false);
    let mut ids: Vec<u32> = Vec::new();
    if add_bos {
        if let Some(bos) = info.bos_token_id {
            ids.push(bos);
        }
    }
    ids.extend_from_slice(&body);
    println!("prompt: {prompt:?}");
    println!("add_bos_token={add_bos} encode_ids={body:?}");
    println!("prompt_ids={ids:?}");
    std::io::stdout().flush().ok();
    if ids_only {
        println!("ids_only: stopping before weight load");
        return Ok(());
    }

    let report = itak_loader::coverage(&file, &cfg);
    report.require_complete().map_err(|e| e.to_string())?;
    println!(
        "coverage: expected={} found={} missing={}",
        report.expected_count(),
        report.found_count(),
        report.missing().len()
    );

    let t_load = Instant::now();
    let weights = itak_loader::load_weights(&file, &cfg).map_err(|e| e.to_string())?;
    println!("phase=load_weights ms={}", t_load.elapsed().as_millis());

    let vocab = info.tokens.clone();
    let model = GlimmerModel::with_diag(
        cfg,
        weights,
        Box::new(cpu_compute::CpuCompute::new()),
        Arc::new(StderrDiag),
    );
    let mut cache = PositionCache::new();

    let mut generated: Vec<u32> = Vec::new();
    let t_prefill = Instant::now();
    let logits = model.forward(&ids, &mut cache).map_err(|e| e.to_string())?;
    let prefill_ms = t_prefill.elapsed().as_millis();
    let top = top_k(&logits, 5);
    println!("phase=prefill ms={prefill_ms} prompt_tokens={}", ids.len());
    print_top("prefill_top", &top, &vocab);
    let mut next = top[0].0;
    generated.push(next);
    println!(
        "gen step=1 id={next} tok={:?} source=prefill",
        vocab[next as usize]
    );
    std::io::stdout().flush().ok();

    for step in 2..=n_tokens {
        let t_step = Instant::now();
        let logits = model
            .forward(&[next], &mut cache)
            .map_err(|e| e.to_string())?;
        let ms = t_step.elapsed().as_millis();
        let top = top_k(&logits, 5);
        print_top(&format!("step{step}_top"), &top, &vocab);
        next = top[0].0;
        generated.push(next);
        println!(
            "gen step={step} id={next} tok={:?} ms={ms}",
            vocab[next as usize]
        );
        std::io::stdout().flush().ok();
    }

    let decoded = tokenizer.decode(&generated).map_err(|e| e.to_string())?;
    println!("generated_ids={generated:?}");
    println!("generated_text={decoded:?}");
    if let Some(kb) = peak_rss_kb() {
        println!("peak_rss_kb={kb}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::top_k;

    #[test]
    fn top_k_orders_by_logit_and_computes_softmax() {
        // Softmax over the full vocabulary, computed by hand:
        // denom = e^-2 + 1 + e^-1 = 1.5032147..., so the top entry
        // holds 1 / denom and the runner-up e^-1 / denom.
        let top = top_k(&[1.0, 3.0, 2.0], 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[0].1, 3.0);
        assert!((top[0].2 - 0.665_241).abs() < 1e-5, "got {}", top[0].2);
        assert_eq!(top[1].0, 2);
        assert!((top[1].2 - 0.244_728).abs() < 1e-5, "got {}", top[1].2);
    }

    #[test]
    fn top_k_tie_breaks_to_the_lowest_id() {
        // Three tokens share the maximum logit. Temperature-0
        // decoding must pick one deterministically: the lowest id,
        // matching a first-max scan, so a run is reproducible.
        let top = top_k(&[2.0, 2.0, 1.0, 2.0], 4);
        let ids: Vec<u32> = top.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(ids, vec![0, 1, 3, 2]);
    }

    #[test]
    fn top_k_greedy_pick_is_the_argmax() {
        let logits = [0.5f32, -1.0, 4.25, 4.0, 0.0];
        let top = top_k(&logits, 5);
        assert_eq!(top[0].0, 2);
        // The full table's probabilities sum to 1.
        let sum: f64 = top.iter().map(|(_, _, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-12, "got {sum}");
    }
}
