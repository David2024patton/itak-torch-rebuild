//! Depth and acceptance sweep for the speculative loop (TODO T-9.7):
//! run the loop once per candidate draft depth against the real
//! Glimmer files, log acceptance per depth, and print the one sweep
//! table the bench note files, ending with the chosen default depth.
//!
//! Usage: `spec_sweep <target.gguf> <draft.gguf> [depths] [n_tokens] [prompt]`
//!
//! `depths` is a comma-separated candidate list (`1,2,4,6`) or the
//! word `default` for the configured candidate set
//! ([`SweepConfig::default`]); `n_tokens` is the per-depth generation
//! budget (default: the configured budget, 128); `prompt` defaults to
//! the Stage 1 prompt. Temperature is the session default (0, greedy):
//! acceptance is a property of the draft and the depth, and at
//! temperature 0 every depth's stream is token-identical anyway (the
//! loop's standing proof, filed in bench/spec-loop-2026-10-30.md).
//!
//! The runner is the library's [`run_sweep`] with a CPU factory: each
//! depth reloads both models, builds a fresh [`SpecSession`], and
//! drops it before the next depth loads, so peak memory is one depth's
//! footprint, not the sweep's. Loads and prefill sit outside the
//! measured wall (the sweep times `generate` only). When the GPU
//! loop lands, its lane wires the same `run_sweep` to a GPU factory
//! and this file's table code does not change.
//!
//! On the CPU reference backend the speculative route is not expected
//! to win (a block-forward row costs roughly 5.5x a single-token step
//! here; see the loop proof note): the sweep's CPU numbers measure
//! acceptance honestly and speed truthfully, and the speed case is
//! re-measured on the GPU path with this same runner.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::rc::Rc;
use std::sync::Arc;

use cpu_compute::CpuCompute;
use dflash::{DflashModel, StderrDiag as DflashStderrDiag, TensorSource as _};
use itak_spec::{
    run_sweep, DflashSpeculator, SharedCompute, SpecConfig, SpecSession, StderrDiag, SweepConfig,
};
use muse_glimmer::GlimmerModel;
use tokenizer::{TokenType, Tokenizer, TokenizerData};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(target_path), Some(draft_path)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: spec_sweep <target.gguf> <draft.gguf> [depths|default] [n_tokens] [prompt]"
        );
        return ExitCode::from(2);
    };
    let default_sweep = SweepConfig::default();
    let depths = match args.get(3).map(String::as_str) {
        None | Some("default") => default_sweep.depths.clone(),
        Some(text) => match SweepConfig::parse_depths(text) {
            Ok(depths) => depths,
            Err(err) => {
                eprintln!("FAIL: {err}");
                return ExitCode::from(2);
            }
        },
    };
    let n_tokens: usize = args.get(4).map_or(default_sweep.tokens_per_depth, |a| {
        a.parse().unwrap_or(default_sweep.tokens_per_depth)
    });
    let prompt = args
        .get(5)
        .map_or("The capital of France is", String::as_str);
    match run(target_path, draft_path, prompt, depths, n_tokens) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("FAIL: {msg}");
            ExitCode::FAILURE
        }
    }
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

/// Build one depth's session over the CPU reference backend: the
/// factory [`run_sweep`] calls per depth. Every load failure is text
/// here (the sweep wraps it with the depth it stopped at); session
/// creation refusals surface through the same channel once built.
fn build_session(
    target_path: &str,
    draft_path: &str,
    eos: &[u32],
    config: &SpecConfig,
) -> Result<SpecSession, String> {
    let draft_file = gguf::GgufFile::open(Path::new(draft_path)).map_err(|e| e.to_string())?;
    let dcfg = itak_loader::load_dflash_config(&draft_file).map_err(|e| e.to_string())?;
    let report = itak_loader::coverage_dflash(&draft_file, &dcfg);
    report.require_complete().map_err(|e| e.to_string())?;
    let dweights =
        itak_loader::load_dflash_weights(&draft_file, &dcfg).map_err(|e| e.to_string())?;
    let draft = DflashModel::with_diag(dcfg, dweights, Arc::new(DflashStderrDiag));

    let target_file = gguf::GgufFile::open(Path::new(target_path)).map_err(|e| e.to_string())?;
    let gcfg = itak_loader::load_config(&target_file).map_err(|e| e.to_string())?;
    let source = itak_loader::DflashTensorSource::new(&target_file);
    let embedding = source
        .tensor("token_embd.weight")
        .map_err(|e| e.to_string())?;
    let output_head = source.tensor("output.weight").map_err(|e| e.to_string())?;
    let gweights = itak_loader::load_weights(&target_file, &gcfg).map_err(|e| e.to_string())?;
    let compute = Rc::new(CpuCompute::new());
    let target = GlimmerModel::new(gcfg, gweights, Box::new(SharedCompute(compute.clone())));
    let speculator = DflashSpeculator::new(draft, embedding, output_head);
    let mut config = config.clone();
    config.eos_tokens = eos.to_vec();
    SpecSession::new(
        target,
        compute,
        Some(speculator),
        config,
        Arc::new(StderrDiag),
    )
    .map_err(|e| e.to_string())
}

fn run(
    target_path: &str,
    draft_path: &str,
    prompt: &str,
    depths: Vec<usize>,
    n_tokens: usize,
) -> Result<(), String> {
    // Tokenize once: the header read is cheap, and every depth must
    // see the identical prompt and stop set.
    let target_file = gguf::GgufFile::open(Path::new(target_path)).map_err(|e| e.to_string())?;
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
    let eos: Vec<u32> = info.eos_token_id.into_iter().collect();
    println!(
        "sweep: prompt {prompt:?} -> ids {prompt_ids:?}; depths {depths:?}; {n_tokens} tokens per depth"
    );
    let _ = std::io::stdout().flush();

    let sweep = SweepConfig {
        depths,
        tokens_per_depth: n_tokens,
    };
    let base = SpecConfig::default();
    let factory = |config: &SpecConfig| build_session(target_path, draft_path, &eos, config);
    let report =
        run_sweep(&sweep, &base, &prompt_ids, factory, &StderrDiag).map_err(|e| e.to_string())?;
    println!();
    print!("{}", report.render_table());
    let _ = std::io::stdout().flush();
    Ok(())
}
