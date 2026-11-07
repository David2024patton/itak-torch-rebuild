//! Real-file proof for the speculative loop on the GPU-resident
//! stack (TODO T-9.6): DFlash drafting against the real Glimmer
//! files with the target verifying on `wgpu-compute`'s resident
//! session, checked against a plain greedy run of the same
//! resident session in the same process.
//!
//! Usage: `spec_generate_gpu <target.gguf> <draft.gguf> [n_tokens] [depth] [prompt] [mode] [stream_file] [--bar=<tok/s>] [--draft=<gpu|cpu>] [--drafter=<dflash|ngram>] [--kv-capacity=<positions>]`
//!
//! `mode` is `both` (default), `plain` (run only phase A and write
//! the stream to `stream_file`), or `spec` (run only phase B,
//! reading the expected stream from `stream_file`). The phases
//! are separate scopes (and can be separate processes) for the
//! reason the CPU runner splits them: each phase's weight set
//! drops before the next builds.
//!
//! Phase A runs the plain greedy baseline on the resident
//! session (prefill plus single-token forwards). Phase B runs a
//! [`SpecSession`] over the resident driver at temperature 0 with
//! the DFlash draft attached. The draft's device path is the
//! `--draft` selection (a [`SpecConfig`] field with the library
//! default CPU): `gpu` (this runner's default, since the runner
//! exists to prove the GPU stack) drives wgpu-compute's resident
//! draft session on the same leaked backend as the target's
//! resident session; `cpu` drives the reference implementation
//! behind the same `Speculator` seam, for A/B runs. The verify
//! side is the resident GPU forward throughout. See
//! `bench/dflash-gpu-2026-11-03.md` for the draft GPU design.
//!
//! `--drafter` selects phase B's draft source (the `SpecMode`
//! name, parsed by `SpecMode::parse`, in the CPU runner
//! `spec_generate`'s style): `dflash` (default) loads the draft
//! file and drafts with the DFlash model on the `--draft`
//! device path; `ngram` never opens the draft file (the
//! positional path is still required; pass any placeholder)
//! and drafts with the n-gram prompt-lookup proposer through
//! `SpecSession::new_ngram` over the same resident target, so
//! the same IDENTITY, GOLDEN, and SPEED verdicts and the same
//! round accounting measure the no-training self-drafting arm
//! against the DFlash arms. `--draft` and `--kv-capacity`
//! configure the DFlash arm only; in ngram mode they are still
//! parsed and validated by name, and the run prints that they
//! do not apply.
//!
//! The T-9.6 bar, printed as verdicts: the speculative stream is
//! token-identical to the plain GPU stream at temperature 0
//! (IDENTITY), the default prompt's first eight tokens match
//! the verified golden continuation (GOLDEN), and the effective
//! speculative rate (tokens emitted during `generate`, divided
//! by its wall time) reaches the recorded llama-server
//! speculative baseline (SPEED; `--bar` overrides the 88.6
//! default for sweeps, TODO T-9.7). The process exit code
//! follows the correctness contract (identity and golden), the
//! CPU runner's convention; the SPEED verdict and the combined
//! T-9.6 line print either way, so a speed miss is never
//! mistaken for a correctness miss or vice versa.
//!
//! Server run (the GPU must be free; the resident session takes
//! the card):
//! `cargo run --release -p itak-spec --example spec_generate_gpu -- /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf /mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf 64 6`
//!
//! The n-gram arm needs no draft file (any placeholder stands
//! in the positional slot):
//! `cargo run --release -p itak-spec --example spec_generate_gpu -- /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf unused.gguf 64 6 "The capital of France is" both plain_stream.txt --drafter=ngram`

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use dflash::{DflashModel, StderrDiag as DflashStderrDiag, TensorSource as _};
use itak_spec::{
    sampler, DflashSpeculator, DraftBackend, NgramConfig, NgramProposer, ResidentTarget,
    SpecConfig, SpecMode, SpecSession, StderrDiag,
};
use muse_glimmer::TensorSource as GlimmerTensorSource;
use tokenizer::{TokenType, Tokenizer, TokenizerData};
use wgpu_compute::{
    DflashRawWeights, DflashSession, DflashSessionConfig, ResidentSession, WgpuCompute,
};

/// The verified greedy continuation of the Stage 1 prompt
/// (bench/gpu-proof-2026-10-30.md): " Paris. It is the most populous".
const GOLDEN_FIRST8: [u32; 8] = [13796, 26, 1573, 373, 290, 2167, 26899, 1212];

/// The T-9.6 speed bar: the recorded llama-server DFlash
/// operating point (107.39 tok/s at depth 6 in the sweep; the
/// TODO item states the bar as the 88.6 baseline figure).
/// An acceptance threshold for this proof, not an operational
/// limit; `--bar` overrides it.
const DEFAULT_BAR_TOK_S: f64 = 88.6;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(target_path), Some(draft_path)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: spec_generate_gpu <target.gguf> <draft.gguf> [n_tokens] [depth] [prompt] \
             [mode] [stream_file] [--bar=<tok/s>] [--drafter=<dflash|ngram>]"
        );
        return ExitCode::from(2);
    };
    let n_tokens: usize = args.get(3).map_or(64, |a| a.parse().unwrap_or(64));
    let depth: usize = args.get(4).map_or(6, |a| a.parse().unwrap_or(6));
    let prompt = args
        .get(5)
        .map_or("The capital of France is", String::as_str);
    let mode = args.get(6).map_or("both", String::as_str);
    let stream_file = args
        .get(7)
        .map_or("plain_stream.txt", String::as_str)
        .to_string();
    let bar: f64 = args
        .iter()
        .find_map(|a| a.strip_prefix("--bar="))
        .map_or(DEFAULT_BAR_TOK_S, |v| {
            v.parse().unwrap_or(DEFAULT_BAR_TOK_S)
        });
    // The draft device path: config with a default (gpu on
    // this runner), never a compiled constant. An unknown
    // value refuses by name instead of guessing.
    let draft_backend = match args.iter().find_map(|a| a.strip_prefix("--draft=")) {
        None => DraftBackend::Gpu,
        Some(text) => match DraftBackend::parse(text) {
            Some(parsed) => parsed,
            None => {
                eprintln!("FAIL: --draft={text} is not a draft backend (expected gpu or cpu)");
                return ExitCode::from(2);
            }
        },
    };
    // The draft source: the CPU runner's `--drafter` flag,
    // parsed by the shared `SpecMode::parse` (no second
    // parser). An unknown or unimplemented value refuses by
    // name before any model loads.
    let drafter = match args.iter().find_map(|a| a.strip_prefix("--drafter=")) {
        Some(text) => match SpecMode::parse(text) {
            Some(m @ (SpecMode::Dflash | SpecMode::Ngram)) => m,
            _ => {
                eprintln!("FAIL: --drafter must be dflash or ngram, got {text:?}");
                return ExitCode::from(2);
            }
        },
        None => SpecMode::Dflash,
    };
    // The GPU draft's resident KV capacity, in positions:
    // session config with the DflashSessionConfig default,
    // raised here for runs whose committed length outgrows
    // it (an injection past capacity refuses by name).
    let kv_capacity: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--kv-capacity="))
        .map_or(DflashSessionConfig::default().kv_capacity, |v| {
            v.parse()
                .unwrap_or(DflashSessionConfig::default().kv_capacity)
        });
    match run(
        target_path,
        draft_path,
        prompt,
        n_tokens,
        depth,
        mode,
        &stream_file,
        bar,
        draft_backend,
        drafter,
        kv_capacity,
    ) {
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

/// One phase's resident session over freshly loaded weights.
/// The session borrows its backend, so the backend handle is
/// leaked deliberately: process-lifetime, one per phase, the
/// runner's own memory discipline (the weights and device
/// state drop with the session at the caller's scope end; the
/// leaked handle is one small allocation per phase).
fn resident_session(
    target_file: &gguf::GgufFile,
    gcfg: &muse_glimmer::GlimmerConfig,
) -> Result<ResidentSession<'static>, String> {
    let backend: &'static WgpuCompute =
        Box::leak(Box::new(WgpuCompute::new().map_err(|e| e.to_string())?));
    resident_session_on(backend, target_file, gcfg)
}

/// A resident session over a caller-supplied backend: phase
/// B shares one leaked backend between the target's resident
/// session and the draft's, so both device residents live on
/// the same device and queue.
fn resident_session_on(
    backend: &'static WgpuCompute,
    target_file: &gguf::GgufFile,
    gcfg: &muse_glimmer::GlimmerConfig,
) -> Result<ResidentSession<'static>, String> {
    let report = itak_loader::coverage(target_file, gcfg);
    println!(
        "target coverage: expected={} found={} missing={}",
        report.expected_count(),
        report.found_count(),
        report.missing().len()
    );
    report.require_complete().map_err(|e| e.to_string())?;
    let weights = itak_loader::load_weights(target_file, gcfg).map_err(|e| e.to_string())?;
    ResidentSession::new(backend, gcfg, &weights).map_err(|e| e.to_string())
}

#[allow(clippy::too_many_arguments)]
fn run(
    target_path: &str,
    draft_path: &str,
    prompt: &str,
    n_tokens: usize,
    depth: usize,
    mode: &str,
    stream_file: &str,
    bar: f64,
    draft_backend: DraftBackend,
    drafter: SpecMode,
    kv_capacity: usize,
) -> Result<bool, String> {
    let total = Instant::now();

    // Load the draft config (coverage gate enforced, dflash
    // arm only) and the target config. The draft WEIGHTS load
    // inside phase B, in the selected arm's own form (f32 for
    // the CPU arm, raw views for the GPU arm), so a GPU run
    // never pays the CPU dequantization and a CPU run never
    // uploads. The n-gram arm never opens the draft file, the
    // CPU runner's precedent: the positional path is still
    // required by the usage line (pass any placeholder), and
    // the run says out loud that the file is not loaded and
    // that `--draft` / `--kv-capacity` do not apply to it.
    let t = Instant::now();
    let (draft_file, dcfg) = if drafter == SpecMode::Ngram {
        println!(
            "drafter: ngram prompt-lookup (draft file {draft_path} not loaded; \
             --draft and --kv-capacity configure the dflash arm only)"
        );
        (None, None)
    } else {
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
        (Some(draft_file), Some(dcfg))
    };

    let target_file = gguf::GgufFile::open(Path::new(target_path)).map_err(|e| e.to_string())?;
    let gcfg = itak_loader::load_config(&target_file).map_err(|e| e.to_string())?;
    if let Some(dcfg) = &dcfg {
        println!(
            "draft: block_size={} target_layers={:?} mask={} | target: layers={} hidden={} vocab={}",
            dcfg.block_size,
            dcfg.target_layers,
            dcfg.mask_token_id,
            gcfg.block_count,
            gcfg.hidden,
            gcfg.vocab_size
        );
    } else {
        println!(
            "drafter: ngram prompt-lookup | target: layers={} hidden={} vocab={}",
            gcfg.block_count, gcfg.hidden, gcfg.vocab_size
        );
    }

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
    phase(
        if drafter == SpecMode::Ngram {
            "load target + tokenizer"
        } else {
            "load draft + tokenizer"
        },
        t,
    );

    // ---- Phase A: plain greedy on the resident session (skipped in
    // spec-only mode, where the expected stream is read from the
    // stream file).
    let t = Instant::now();
    let plain_stream = if mode == "spec" {
        let text = std::fs::read_to_string(stream_file).map_err(|e| e.to_string())?;
        let ids: Vec<u32> = text
            .split_whitespace()
            .map(|w| {
                w.parse()
                    .map_err(|_| format!("bad token id {w:?} in {stream_file}"))
            })
            .collect::<Result<_, _>>()?;
        println!("expected stream read from {stream_file}: {ids:?}");
        ids
    } else {
        let mut session = resident_session(&target_file, &gcfg)?;
        let tp = Instant::now();
        let mut logits = session.forward(&prompt_ids).map_err(|e| e.to_string())?;
        let prefill_s = tp.elapsed().as_secs_f64();
        println!("plain prefill (resident): {prefill_s:.3}s");
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
                logits = session.forward(&[token]).map_err(|e| e.to_string())?;
            }
        }
        let decode_s = td.elapsed().as_secs_f64();
        let stats = session.stats();
        println!(
            "plain stream: {stream:?} ({:.2} tok/s decode, {} chunks, {} submissions, \
             {} dispatches)",
            stream.len() as f64 / decode_s.max(f64::MIN_POSITIVE),
            stats.chunks,
            stats.submissions,
            stats.dispatches
        );
        let text: String = stream.iter().map(|id| format!("{id} ")).collect();
        std::fs::write(stream_file, text).map_err(|e| e.to_string())?;
        println!("plain stream written to {stream_file}");
        stream
    };
    let plain_wall = t.elapsed().as_secs_f64();
    phase("phase A plain greedy (resident)", t);
    if mode == "plain" {
        println!("PLAIN ONLY: phase B skipped by mode");
        phase("total", total);
        return Ok(true);
    }

    // ---- Phase B: the speculative session over the resident driver,
    // temperature 0. The dflash arm drafts with the DFlash model on
    // the selected device path behind the Speculator seam; the
    // n-gram arm drafts with the prompt-lookup proposer through
    // `SpecSession::new_ngram` over the same resident target, with
    // no draft model built at all.
    let t = Instant::now();
    let spec_stream = {
        // One leaked backend serves both residents (the
        // runner's memory discipline, as in phase A).
        let backend: &'static WgpuCompute =
            Box::leak(Box::new(WgpuCompute::new().map_err(|e| e.to_string())?));
        let session = resident_session_on(backend, &target_file, &gcfg)?;
        let (mut session, prefill_kind) = match drafter {
            SpecMode::Ngram => {
                let proposer = NgramProposer::new(NgramConfig::default(), gcfg.vocab_size as usize)
                    .map_err(|e| e.to_string())?;
                let config = SpecConfig {
                    mode: SpecMode::Ngram,
                    depth,
                    eos_tokens: eos,
                    ..SpecConfig::default()
                };
                println!("drafter: ngram prompt-lookup (default proposer config)");
                let session = SpecSession::new_ngram(
                    ResidentTarget::new(session),
                    Some(proposer),
                    config,
                    Arc::new(StderrDiag),
                )
                .map_err(|e| e.to_string())?;
                (session, "plain + proposer commit")
            }
            _ => {
                let draft_file = draft_file
                    .as_ref()
                    .expect("the dflash arm loaded the draft file");
                let dcfg = dcfg
                    .as_ref()
                    .expect("the dflash arm loaded the draft config");
                let speculator = match draft_backend {
                    DraftBackend::Cpu => {
                        let dweights = itak_loader::load_dflash_weights(draft_file, dcfg)
                            .map_err(|e| e.to_string())?;
                        let draft = DflashModel::with_diag(
                            dcfg.clone(),
                            dweights,
                            Arc::new(DflashStderrDiag),
                        );
                        let source = itak_loader::DflashTensorSource::new(&target_file);
                        let embedding = source
                            .tensor("token_embd.weight")
                            .map_err(|e| e.to_string())?;
                        let output_head =
                            source.tensor("output.weight").map_err(|e| e.to_string())?;
                        println!("draft backend: cpu (reference implementation)");
                        DflashSpeculator::new(draft, embedding, output_head)
                    }
                    DraftBackend::Gpu => {
                        let draft_source = itak_loader::GgufTensorSource::new(draft_file);
                        let raw = DflashRawWeights::load(&draft_source, dcfg)
                            .map_err(|e| e.to_string())?;
                        let target_source = itak_loader::GgufTensorSource::new(&target_file);
                        let embedding =
                            GlimmerTensorSource::tensor(&target_source, "token_embd.weight")
                                .map_err(|e| e.to_string())?;
                        let output_head =
                            GlimmerTensorSource::tensor(&target_source, "output.weight")
                                .map_err(|e| e.to_string())?;
                        let draft_session = DflashSession::with_diag(
                            backend,
                            dcfg,
                            &raw,
                            Some(&embedding),
                            Some(&output_head),
                            DflashSessionConfig { kv_capacity },
                            Arc::new(DflashStderrDiag),
                        )
                        .map_err(|e| e.to_string())?;
                        println!(
                            "draft backend: gpu (resident draft session, \
                             kv_capacity={kv_capacity})"
                        );
                        DflashSpeculator::gpu(draft_session)
                    }
                };
                let config = SpecConfig {
                    depth,
                    draft_backend,
                    eos_tokens: eos,
                    ..SpecConfig::default()
                };
                let session = SpecSession::new(
                    ResidentTarget::new(session),
                    Some(speculator),
                    config,
                    Arc::new(StderrDiag),
                )
                .map_err(|e| e.to_string())?;
                (session, "capturing")
            }
        };
        let tp = Instant::now();
        let pending = session.begin(&prompt_ids).map_err(|e| e.to_string())?;
        let prefill_s = tp.elapsed().as_secs_f64();
        println!("spec prefill (resident, {prefill_kind}): {prefill_s:.3}s, pending={pending}");
        let _ = std::io::stdout().flush();
        let tg = Instant::now();
        let stream = session.generate(n_tokens, &[]).map_err(|e| e.to_string())?;
        let generate_s = tg.elapsed().as_secs_f64();
        let resident_stats = session.driver().session().stats();
        let stats = session.finish();
        // Effective rate: tokens the generate phase emitted past
        // the prefill's pending token, over the generate wall.
        let effective = (stream.len() - 1) as f64 / generate_s.max(f64::MIN_POSITIVE);
        let all_in = stream.len() as f64 / (prefill_s + generate_s).max(f64::MIN_POSITIVE);
        println!("spec stream: {stream:?}");
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
            "spec resident counters: forwards={} chunks={} submissions={} readbacks={} \
             dispatches={}",
            resident_stats.forwards,
            resident_stats.chunks,
            resident_stats.submissions,
            resident_stats.readbacks,
            resident_stats.dispatches
        );
        println!(
            "spec rates: effective {effective:.2} tok/s (generate phase), all-in {all_in:.2} \
             tok/s (prefill + generate)"
        );
        (stream, effective)
    };
    let (spec_stream, effective_tok_s) = spec_stream;
    let spec_wall = t.elapsed().as_secs_f64();
    println!(
        "wall: plain {plain_wall:.1}s total, spec {spec_wall:.1}s total (phase walls include \
         weight load and session build)"
    );
    phase("phase B speculative (resident)", t);

    // ---- The verdicts.
    let mut ok = true;
    let identity = spec_stream == plain_stream;
    if identity {
        println!(
            "IDENTITY: PASS ({} tokens token-identical to plain GPU greedy)",
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
    let mut golden = true;
    if prompt == "The capital of France is" && spec_stream.len() >= 8 {
        if spec_stream[..8] == GOLDEN_FIRST8 {
            println!("GOLDEN: PASS (first 8 match the verified continuation)");
        } else {
            golden = false;
            ok = false;
            println!(
                "GOLDEN: FAIL (first 8 are {:?}, expected {GOLDEN_FIRST8:?})",
                &spec_stream[..8]
            );
        }
    }
    let speed = effective_tok_s >= bar;
    if speed {
        println!("SPEED: PASS (effective {effective_tok_s:.2} tok/s >= bar {bar:.1})");
    } else {
        println!("SPEED: FAIL (effective {effective_tok_s:.2} tok/s < bar {bar:.1})");
    }
    println!(
        "T-9.6: {}",
        if identity && golden && speed {
            "PASS (identity, golden, and speed bars all met)"
        } else if identity && golden {
            "CORRECTNESS PASS, SPEED FAIL (see the SPEED line above and \
             bench/dflash-gpu-2026-11-03.md)"
        } else {
            "FAIL (correctness bar missed; see the verdicts above)"
        }
    );
    phase("total", total);
    Ok(ok)
}
