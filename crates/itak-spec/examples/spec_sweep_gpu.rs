//! Depth and acceptance sweep for the speculative loop on the
//! GPU-resident stack (TODO T-9.7): the library's [`run_sweep`]
//! wired to a factory that builds each depth's session over
//! `wgpu-compute`'s resident target session with the DFlash
//! draft on its GPU resident path (the `spec_generate_gpu`
//! runner's phase B wiring), so the sweep measures the GPU
//! loop's real depth curve instead of the CPU reference's.
//!
//! Usage: `spec_sweep_gpu <target.gguf> <draft.gguf> [depths|default] [n_tokens] [prompt]`
//!
//! `depths` is a comma-separated candidate list (`1,2,4,6`) or
//! the word `default` for the configured candidate set
//! ([`SweepConfig::default`]: [1, 2, 3, 4, 6, 8, 12, 16]);
//! `n_tokens` is the per-depth generation budget (default: the
//! configured budget, 128); `prompt` defaults to the Stage 1
//! prompt. Temperature is 0 (greedy), as in the CPU sweep.
//!
//! Each depth reloads both models and builds a fresh session
//! (the CPU example's discipline): peak memory is one depth's
//! footprint. Loads and prefill sit outside the measured wall
//! (`run_sweep` times `generate` only). Server run, GPU free:
//!
//! `cargo run --release -p itak-spec --example spec_sweep_gpu -- /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf /mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf`
//!
//! NOTE (server proof, 2026-11-04): `run_sweep` as merged was
//! typed to the default `SpecSession` (CPU driver). Wiring it
//! to the resident factory required widening `run_sweep` and
//! its `measure_depth` helper in `crates/itak-spec/src/sweep.rs`
//! to be generic over the session's `TargetDriver` (a two-line
//! signature change; the CPU callers infer the CPU driver
//! unchanged). That widening is part of this lane's patch set
//! and must land with this example.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use dflash::StderrDiag as DflashStderrDiag;
use itak_spec::{
    run_sweep, DflashSpeculator, DraftBackend, ResidentTarget, SpecConfig, SpecSession, StderrDiag,
    SweepConfig,
};
use muse_glimmer::TensorSource as GlimmerTensorSource;
use tokenizer::{TokenType, Tokenizer, TokenizerData};
use wgpu_compute::{
    DflashRawWeights, DflashSession, DflashSessionConfig, ResidentSession, WgpuCompute,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(target_path), Some(draft_path)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: spec_sweep_gpu <target.gguf> <draft.gguf> [depths|default] [n_tokens] [prompt]"
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

/// Build one depth's session over the GPU-resident stack: the
/// factory [`run_sweep`] calls per depth. Mirrors
/// `spec_generate_gpu`'s phase B exactly: one leaked backend
/// serves both residents, the draft loads raw (no CPU
/// dequantization), and the target resident session is built
/// through the loader coverage gate. The backend handle leaks
/// per depth (one small allocation per depth, process-lifetime;
/// the sessions and their device state drop between depths).
fn build_session(
    target_path: &str,
    draft_path: &str,
    eos: &[u32],
    config: &SpecConfig,
) -> Result<SpecSession<ResidentTarget<'static>>, String> {
    let draft_file = gguf::GgufFile::open(Path::new(draft_path)).map_err(|e| e.to_string())?;
    let dcfg = itak_loader::load_dflash_config(&draft_file).map_err(|e| e.to_string())?;
    let report = itak_loader::coverage_dflash(&draft_file, &dcfg);
    report.require_complete().map_err(|e| e.to_string())?;

    let target_file = gguf::GgufFile::open(Path::new(target_path)).map_err(|e| e.to_string())?;
    let gcfg = itak_loader::load_config(&target_file).map_err(|e| e.to_string())?;
    let report = itak_loader::coverage(&target_file, &gcfg);
    report.require_complete().map_err(|e| e.to_string())?;

    let backend: &'static WgpuCompute =
        Box::leak(Box::new(WgpuCompute::new().map_err(|e| e.to_string())?));

    // The GPU draft arm (the runner's default selection).
    let draft_source = itak_loader::GgufTensorSource::new(&draft_file);
    let raw = DflashRawWeights::load(&draft_source, &dcfg).map_err(|e| e.to_string())?;
    let target_source = itak_loader::GgufTensorSource::new(&target_file);
    let embedding = GlimmerTensorSource::tensor(&target_source, "token_embd.weight")
        .map_err(|e| e.to_string())?;
    let output_head =
        GlimmerTensorSource::tensor(&target_source, "output.weight").map_err(|e| e.to_string())?;
    let draft_session = DflashSession::with_diag(
        backend,
        &dcfg,
        &raw,
        Some(&embedding),
        Some(&output_head),
        DflashSessionConfig::default(),
        Arc::new(DflashStderrDiag),
    )
    .map_err(|e| e.to_string())?;
    let speculator = DflashSpeculator::gpu(draft_session);

    // The resident target on the same backend.
    let weights = itak_loader::load_weights(&target_file, &gcfg).map_err(|e| e.to_string())?;
    let session = ResidentSession::new(backend, &gcfg, &weights).map_err(|e| e.to_string())?;

    let mut config = config.clone();
    config.draft_backend = DraftBackend::Gpu;
    config.eos_tokens = eos.to_vec();
    SpecSession::new(
        ResidentTarget::new(session),
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
    // Tokenize once: every depth sees the identical prompt and
    // stop set (the CPU sweep example's discipline).
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
        "sweep (gpu): prompt {prompt:?} -> ids {prompt_ids:?}; depths {depths:?}; {n_tokens} tokens per depth"
    );
    let _ = std::io::stdout().flush();

    let sweep = SweepConfig {
        depths,
        tokens_per_depth: n_tokens,
        ..SweepConfig::default()
    };
    let base = SpecConfig {
        draft_backend: DraftBackend::Gpu,
        ..SpecConfig::default()
    };
    let factory = |config: &SpecConfig| build_session(target_path, draft_path, &eos, config);
    let report =
        run_sweep(&sweep, &base, &prompt_ids, factory, &StderrDiag).map_err(|e| e.to_string())?;
    println!();
    print!("{}", report.render_table());
    let _ = std::io::stdout().flush();
    Ok(())
}
