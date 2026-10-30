//! Real-file proof for the speculative loop (SPEC-itak-spec Sections 4
//! and 9): DFlash drafting against the real Glimmer files on CPU, with
//! the token-identity guarantee checked against a plain greedy run of
//! the same target in the same process.
//!
//! Usage: `spec_generate <target.gguf> <draft.gguf> [n_tokens] [depth] [prompt] [mode] [stream_file]`
//!
//! `mode` is `both` (default), `plain` (run only phase A and write the
//! stream to `stream_file`), or `spec` (run only phase B, reading the
//! expected stream from `stream_file`). Splitting the phases into two
//! processes halves the peak memory: this runner's first server attempt
//! was OOM-killed with both phases' weight sets resident.
//!
//! Phase A runs the plain greedy baseline (prefill plus single-token
//! steps). Phase B runs a [`SpecSession`] at temperature 0 with the
//! draft attached. The streams must be token-identical; the first eight
//! tokens of the default Stage 1 prompt must equal the verified golden
//! continuation. Wall times per token are printed for both routes: on
//! CPU the speculative route is not expected to win (the mechanism is
//! the deliverable; speed belongs to the GPU path), and the numbers are
//! reported as measured either way.
//!
//! This is a correctness runner, not a benchmark: the CPU reference
//! dequantizes as it goes, so both phases are minutes-to-hours class.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use cpu_compute::CpuCompute;
use dflash::{DflashModel, TensorSource as _};
use itak_spec::{sampler, DflashSpeculator, SharedCompute, SpecConfig, SpecSession, StderrDiag};
use muse_glimmer::{GlimmerModel, PositionCache};
use tokenizer::{TokenType, Tokenizer, TokenizerData};

/// The verified greedy continuation of the Stage 1 prompt
/// (bench/gpu-proof-2026-10-30.md): " Paris. It is the most populous".
const GOLDEN_FIRST8: [u32; 8] = [13796, 26, 1573, 373, 290, 2167, 26899, 1212];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(target_path), Some(draft_path)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: spec_generate <target.gguf> <draft.gguf> [n_tokens] [depth] [prompt]");
        return ExitCode::from(2);
    };
    let n_tokens: usize = args.get(3).map_or(16, |a| a.parse().unwrap_or(16));
    let depth: usize = args.get(4).map_or(6, |a| a.parse().unwrap_or(6));
    let prompt = args
        .get(5)
        .map_or("The capital of France is", String::as_str);
    let mode = args.get(6).map_or("both", String::as_str);
    let stream_file = args
        .get(7)
        .map_or("plain_stream.txt", String::as_str)
        .to_string();
    match run(target_path, draft_path, prompt, n_tokens, depth, mode, &stream_file) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(msg) => {
            eprintln!("FAIL: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn phase(label: &str, t: Instant) {
    println!("[{label}] {:.1}s", t.elapsed().as_secs_f64());
    let _ = std::io::stdout().flush();
}

/// Same thin tokenizer adapter as the other integration runners.
fn data_from_info(
    info: &gguf::TokenizerInfo,
    model: Option<String>,
    unknown: Option<u32>,
) -> TokenizerData {
    TokenizerData {
        model,
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
    }
}

fn run(
    target_path: &str,
    draft_path: &str,
    prompt: &str,
    n_tokens: usize,
    depth: usize,
    mode: &str,
    stream_file: &str,
) -> Result<bool, String> {
    let total = Instant::now();

    // Load the draft (coverage gate enforced) and the target.
    let t = Instant::now();
    let draft_file = gguf::GgufFile::open(Path::new(draft_path)).map_err(|e| e.to_string())?;
    let dcfg = itak_loader::load_dflash_config(&draft_file).map_err(|e| e.to_string())?;
    let report = itak_loader::coverage_dflash(&draft_file, &dcfg);
    println!(
        "draft coverage: expected={} found={} missing={}",
        report.expected_count(),
        report.found_count(),
        report.missing().len()
    );
    report.require_complete().map_err(|e| e.to_string())?;
    let dweights =
        itak_loader::load_dflash_weights(&draft_file, &dcfg).map_err(|e| e.to_string())?;
    let draft = DflashModel::new(dcfg.clone(), dweights);

    let target_file = gguf::GgufFile::open(Path::new(target_path)).map_err(|e| e.to_string())?;
    let gcfg = itak_loader::load_config(&target_file).map_err(|e| e.to_string())?;
    println!(
        "draft: block_size={} target_layers={:?} mask={} | target: layers={} hidden={} vocab={}",
        dcfg.block_size, dcfg.target_layers, dcfg.mask_token_id, gcfg.block_count, gcfg.hidden,
        gcfg.vocab_size
    );

    let info = target_file.tokenizer().map_err(|e| e.to_string())?;
    let tok_model = target_file
        .metadata()
        .opt_string("tokenizer.ggml.model")
        .map_err(|e| e.to_string())?
        .map(str::to_string);
    let tokenizer =
        Tokenizer::new(data_from_info(&info, tok_model, None)).map_err(|e| e.to_string())?;
    let mut prompt_ids = tokenizer.encode(prompt).map_err(|e| e.to_string())?;
    let add_bos = target_file
        .metadata()
        .get_bool("tokenizer.ggml.add_bos_token")
        .unwrap_or(false);
    if add_bos {
        if let Some(bos) = info.bos_token_id {
            prompt_ids.insert(0, bos);
        }
    }
    println!("prompt {prompt:?} -> ids {prompt_ids:?}");
    let eos: Vec<u32> = info.eos_token_id.into_iter().collect();
    phase("load models + tokenizer", t);

    // ---- Phase A: plain greedy baseline (skipped in spec-only mode,
    // where the expected stream is read from the stream file).
    let t = Instant::now();
    let plain_stream = if mode == "spec" {
        let text = std::fs::read_to_string(stream_file).map_err(|e| e.to_string())?;
        let ids: Vec<u32> = text
            .split_whitespace()
            .map(|w| w.parse().map_err(|_| format!("bad token id {w:?} in {stream_file}")))
            .collect::<Result<_, _>>()?;
        println!("expected stream read from {stream_file}: {ids:?}");
        ids
    } else {
    let gweights = itak_loader::load_weights(&target_file, &gcfg).map_err(|e| e.to_string())?;
    let stream = {
        let target = GlimmerModel::new(gcfg.clone(), gweights, Box::new(CpuCompute::new()));
        let mut cache = PositionCache::new();
        let tp = Instant::now();
        let mut logits = target
            .forward(&prompt_ids, &mut cache)
            .map_err(|e| e.to_string())?;
        println!("plain prefill: {:.1}s", tp.elapsed().as_secs_f64());
        let _ = std::io::stdout().flush();
        let td = Instant::now();
        let mut stream = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let token = sampler::argmax(&logits);
            stream.push(token);
            if eos.contains(&token) {
                println!("plain: EOS at generated position {i}");
                break;
            }
            if i + 1 < n_tokens {
                logits = target
                    .forward(&[token], &mut cache)
                    .map_err(|e| e.to_string())?;
            }
        }
        let decode_s = td.elapsed().as_secs_f64();
        println!(
            "plain stream: {stream:?} ({:.1}s decode, {:.2}s/token after prefill)",
            decode_s,
            decode_s / stream.len().max(1) as f64
        );
        stream
    };
    let text: String = stream.iter().map(|id| format!("{id} ")).collect();
    std::fs::write(stream_file, text).map_err(|e| e.to_string())?;
    println!("plain stream written to {stream_file}");
    stream
    };
    let plain_wall = t.elapsed().as_secs_f64();
    phase("phase A plain greedy", t);
    if mode == "plain" {
        println!("PLAIN ONLY: phase B skipped by mode");
        phase("total", total);
        return Ok(true);
    }

    // ---- Phase B: speculative session, temperature 0.
    let t = Instant::now();
    let source = itak_loader::DflashTensorSource::new(&target_file);
    let embedding = source
        .tensor("token_embd.weight")
        .map_err(|e| e.to_string())?;
    let output_head = source.tensor("output.weight").map_err(|e| e.to_string())?;
    let gweights = itak_loader::load_weights(&target_file, &gcfg).map_err(|e| e.to_string())?;
    let compute = Rc::new(CpuCompute::new());
    let target = GlimmerModel::new(
        gcfg.clone(),
        gweights,
        Box::new(SharedCompute(compute.clone())),
    );
    let speculator = DflashSpeculator::new(draft, embedding, output_head);
    let config = SpecConfig {
        depth,
        eos_tokens: eos,
        ..SpecConfig::default()
    };
    let mut session = SpecSession::new(
        target,
        compute,
        Some(speculator),
        config,
        Arc::new(StderrDiag),
    )
    .map_err(|e| e.to_string())?;
    let tp = Instant::now();
    let pending = session.begin(&prompt_ids).map_err(|e| e.to_string())?;
    println!(
        "spec prefill (capturing): {:.1}s, pending={pending}",
        tp.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();
    let spec_stream = session
        .generate(n_tokens, &[])
        .map_err(|e| e.to_string())?;
    let stats = session.finish();
    let spec_wall = t.elapsed().as_secs_f64();
    println!("spec stream: {spec_stream:?}");
    println!(
        "spec stats: rounds={} proposed={} accepted={} acceptance_rate={:.3} \
         rejections={} rollbacks={} tokens_per_round={:.2} fallback_latched={}",
        stats.rounds,
        stats.proposed,
        stats.accepted,
        stats.acceptance_rate(),
        stats.rejections,
        stats.rollbacks,
        stats.tokens_per_round(),
        stats.fallback_latched
    );
    println!(
        "wall: plain {:.1}s total ({:.2}s/token), spec {:.1}s total ({:.2}s/token)",
        plain_wall,
        plain_wall / plain_stream.len().max(1) as f64,
        spec_wall,
        spec_wall / spec_stream.len().max(1) as f64
    );
    phase("phase B speculative", t);

    // ---- The verdict.
    let mut ok = true;
    if spec_stream == plain_stream {
        println!(
            "IDENTITY: PASS ({} tokens token-identical to plain greedy)",
            spec_stream.len()
        );
    } else {
        ok = false;
        println!("IDENTITY: FAIL");
        println!("  plain: {plain_stream:?}");
        println!("  spec:  {spec_stream:?}");
        for (i, (a, b)) in plain_stream.iter().zip(spec_stream.iter()).enumerate() {
            if a != b {
                println!("  first divergence at generated position {i}: plain {a} vs spec {b}");
                break;
            }
        }
    }
    if prompt == "The capital of France is" && spec_stream.len() >= 8 {
        if spec_stream[..8] == GOLDEN_FIRST8 {
            println!("GOLDEN: PASS (first 8 match the verified continuation)");
        } else {
            ok = false;
            println!(
                "GOLDEN: FAIL (first 8 are {:?}, expected {GOLDEN_FIRST8:?})",
                &spec_stream[..8]
            );
        }
    }
    phase("total", total);
    Ok(ok)
}
