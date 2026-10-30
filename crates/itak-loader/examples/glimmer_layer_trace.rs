//! Per-layer activation trace for the muse-glimmer forward: the
//! localization tool for Integration Run B's first-token divergence
//! (see INTEGRATION-RUN-B.md and FORWARD-DIFF-PLAN.md in this
//! crate).
//!
//! Usage: `glimmer_layer_trace <path-to-gguf> [prompt]
//! [--first-n N] [--out <path>] [--ids-only]`
//!
//! The prompt defaults to the golden `factual_capital` prompt,
//! "The capital of France is", tokenized exactly as Run B tokenized
//! it (BOS prepended per the file's own metadata), so the ids are
//! the byte-identical input both engines consumed. The run is one
//! traced prefill over those ids: no decode steps, because the
//! divergence Run B measured is already total at prefill position 1.
//!
//! Output is the stable trace line format (version 1, documented in
//! the `muse_glimmer::trace` module): header lines naming the tool,
//! the model shape, and the prompt ids, then one line per stage
//! (`embed`, `layer.<i>.mid`, `layer.<i>.out`, `final_norm`,
//! `logits`) carrying whole-tensor and last-row statistics, then one
//! dimension line per stage (version 1, documented in the
//! `muse_glimmer::dimension` module: participation ratio, plus the
//! TwoNN estimate when the prompt has enough positions for it to
//! run), then the prefill top-5 in Run B's table format for a direct
//! cross-check against its position-1 numbers. `--out` writes the
//! trace and dimension lines to a file instead of stdout so the dump
//! can be committed next to a reference dump and diffed.
//! `--ids-only` stops after printing the prompt ids, before any
//! weight work.
//!
//! This is a correctness tool, not a benchmark: one 30B prefill on
//! `cpu-compute` is minutes-class, exactly as Run B measured.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use muse_glimmer::{
    DimensionConfig, DimensionLog, DimensionProbe, GlimmerModel, PositionCache, StderrDiag,
    TraceLog, TraceSink,
};
use tokenizer::{TokenType, Tokenizer, TokenizerData};

/// The prompt ids Integration Run B fed both engines, from
/// INTEGRATION-RUN-B.md. The dump is only comparable to Run B (and
/// to a reference dump) when the ids match these.
const RUN_B_IDS: [u32; 6] = [200000, 954, 7963, 323, 11698, 373];

/// The golden `factual_capital` prompt (golden/prompts.jsonl).
const GOLDEN_PROMPT: &str = "The capital of France is";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!(
            "usage: glimmer_layer_trace <path-to-gguf> [prompt] [--first-n N] [--out <path>] [--ids-only]"
        );
        return ExitCode::from(2);
    };
    let prompt = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .map_or(GOLDEN_PROMPT, String::as_str);
    let first_n: usize = parse_flag(&args, "--first-n").map_or(8, |v| v.parse().unwrap_or(8));
    let out = parse_flag(&args, "--out");
    let ids_only = args.iter().any(|a| a == "--ids-only");
    match run(path, prompt, first_n, out, ids_only) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("FAIL: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// The value following `--name` in the argument list, if present.
fn parse_flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
}

/// Adapts the gguf crate's parsed tokenizer info into the tokenizer
/// crate's input data. Same thin adapter as the other runners in
/// this crate; it lives in harness code, not in either library.
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
/// the whole vocabulary), best first. Same helper as
/// glimmer_generate, so the top-5 prints are directly comparable.
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
    });
    idx.truncate(k);
    idx.iter()
        .map(|&id| {
            let z = logits[id as usize];
            (id, z, f64::from(z - max).exp() / denom)
        })
        .collect()
}

/// Forwards every capture to both the trace log and the dimension
/// log, so one traced prefill produces both dumps. Harness glue in
/// the same spirit as `data_from_info`: the two sinks stay separate
/// library modules; the tee lives here.
struct TeeTrace<'a> {
    trace: &'a mut TraceLog,
    dims: &'a mut DimensionLog,
}

impl TraceSink for TeeTrace<'_> {
    fn capture(&mut self, stage: &str, tensor: &[f32], rows: usize, cols: usize) {
        self.trace.capture(stage, tensor, rows, cols);
        self.dims.capture(stage, tensor, rows, cols);
    }
}

fn run(
    path: &str,
    prompt: &str,
    first_n: usize,
    out: Option<&str>,
    ids_only: bool,
) -> Result<(), String> {
    let file = gguf::GgufFile::open(Path::new(path)).map_err(|e| e.to_string())?;
    println!("trace_header v=1 tool=glimmer_layer_trace");
    println!("file: {path} bytes={}", file.file_len());

    let cfg = itak_loader::load_config(&file).map_err(|e| e.to_string())?;
    println!(
        "config: layers={} hidden={} vocab={}",
        cfg.block_count, cfg.hidden, cfg.vocab_size
    );

    // Tokenize before any weight work, exactly as Run B did: the
    // printed ids are the contract the reference dump must match.
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
    println!("prompt_ids={ids:?}");
    println!("run_b_ids_match={}", ids.as_slice() == RUN_B_IDS);
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
    let mut log = TraceLog::new(first_n);
    // The dimension probe shares the model's diagnostic channel, so
    // a stage whose TwoNN gate refuses (the default prompt has 6
    // positions, below the 32-point minimum) says so on stderr
    // beside the run's other refusals, and its dim line shows
    // twonn=na.
    let mut dims = DimensionLog::new(DimensionProbe::with_diag(
        DimensionConfig::default(),
        Arc::new(StderrDiag),
    ));
    let mut tee = TeeTrace {
        trace: &mut log,
        dims: &mut dims,
    };

    let t_prefill = Instant::now();
    let logits = model
        .forward_traced(&ids, &mut cache, &mut tee)
        .map_err(|e| e.to_string())?;
    println!(
        "phase=traced_prefill ms={} prompt_tokens={} stages={}",
        t_prefill.elapsed().as_millis(),
        ids.len(),
        log.records().len()
    );

    let mut lines = log.lines();
    lines.extend(dims.lines());
    match out {
        Some(out_path) => {
            let mut text = lines.join("\n");
            text.push('\n');
            std::fs::write(out_path, text).map_err(|e| e.to_string())?;
            println!("trace_out={out_path} lines={}", lines.len());
        }
        None => {
            for line in &lines {
                println!("{line}");
            }
        }
    }
    std::io::stdout().flush().ok();

    // Cross-check against Run B's position-1 table: the traced
    // prefill must reproduce its top-5 exactly, or the trace itself
    // is not measuring the same forward.
    for (rank, (id, logit, prob)) in top_k(&logits, 5).iter().enumerate() {
        println!(
            "prefill_top rank={rank} id={id} tok={:?} logit={logit:.6} prob={prob:.6}",
            vocab[*id as usize]
        );
    }
    std::io::stdout().flush().ok();
    Ok(())
}
