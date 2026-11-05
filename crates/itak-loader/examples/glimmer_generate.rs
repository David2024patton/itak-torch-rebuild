//! Integration run B: Glimmer's first generation on the new Rust
//! stack (TODO: integration stage), driven end to end through the
//! real crates: `gguf` parses the file, `itak-loader` loads config
//! and weights behind the coverage gate, the `tokenizer` crate
//! tokenizes, and `muse-glimmer`'s forward runs on `cpu-compute`.
//!
//! Usage: `glimmer_generate <path-to-gguf> <prompt> [n_tokens] [--ids-only] [--gpu|--gpu-trait|--gpu-argmax]`
//!
//! `--gpu` runs the device-resident session in `wgpu-compute`:
//! weights upload once, activations and KV never leave the
//! card, and each forward chunk is one command-buffer chain
//! with only the finished logits read back (the fast path; see
//! `wgpu-compute`'s session module). `--gpu-argmax` runs the
//! same resident session with the on-device argmax tail: each
//! chunk reads back eight bytes (the winning index and its
//! logit, reduced on the GPU) instead of the logits vector,
//! so its per-step lines carry the winner and its raw logit
//! rather than a top-5. `--gpu-trait` selects the older trait
//! path over the same backend (per-step host round trips
//! through `GlimmerModel`), kept as the reference and for
//! comparison runs. The default is the `cpu-compute`
//! reference. Same model, same output contract on all paths.
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

/// EXPERIMENT ONLY (overnight logit-scale run, 2026-11-05): the
/// environment variable `ITAK_EXPERIMENT_LOGIT_SCALE`, when set to a
/// positive finite number, replaces the file's `logit_scale` in the
/// config this example hands to the model. Unset (the default) the
/// file value is used untouched. This exists to measure how sensitive
/// Glimmer's output distribution is to the scale (TODO T-9.3 residue);
/// it is not a shipped knob and no library code reads it.
fn experiment_logit_scale(raw: Option<&str>, file_value: f32) -> Result<f32, String> {
    let Some(raw) = raw else {
        return Ok(file_value);
    };
    let forced: f32 = raw
        .parse()
        .map_err(|_| format!("ITAK_EXPERIMENT_LOGIT_SCALE: not a number: {raw:?}"))?;
    if !forced.is_finite() || forced <= 0.0 {
        return Err(format!(
            "ITAK_EXPERIMENT_LOGIT_SCALE: must be positive and finite, got {forced}"
        ));
    }
    Ok(forced)
}

/// Diagnostics on one finished (scaled, softcapped) logit vector:
/// the finished range, the implied pre-softcap range recovered by
/// inverting `cap * tanh(w / cap)` (entries saturated to the cap in
/// f32 are counted, and the recovered max is then a lower bound),
/// the implied raw lm_head range (pre-softcap divided by the scale),
/// and how many of the top 100 finished logits sit beyond
/// `0.9 * cap`, the saturation yardstick of the experiment note.
fn print_logit_stats(label: &str, logits: &[f32], cfg: &muse_glimmer::GlimmerConfig) {
    let cap = cfg.logit_softcap;
    let scale = cfg.logit_scale;
    let fin_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let fin_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top100 = &sorted[..sorted.len().min(100)];
    let beyond = top100.iter().filter(|&&z| z > 0.9 * cap).count();
    let saturated = logits.iter().filter(|&&z| z >= cap).count();
    // Invert the softcap on the entries that did not saturate to the
    // cap in f32; the recovered range is over those entries only.
    let mut pre_min = f32::INFINITY;
    let mut pre_max = f32::NEG_INFINITY;
    for &z in logits {
        if z < cap && z > -cap {
            let w = cap * (z / cap).atanh();
            pre_min = pre_min.min(w);
            pre_max = pre_max.max(w);
        }
    }
    println!(
        "{label} stats: finished_range=[{fin_min:.4},{fin_max:.4}] \
         presoftcap_range=[{pre_min:.4},{pre_max:.4}] saturated_to_cap={saturated} \
         raw_range=[{:.4},{:.4}] top100_beyond_0.9cap={beyond}",
        pre_min / scale,
        pre_max / scale,
    );
    std::io::stdout().flush().ok();
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(path), Some(prompt)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: glimmer_generate <path-to-gguf> <prompt> [n_tokens] [--ids-only] \
             [--gpu|--gpu-trait|--gpu-argmax]"
        );
        return ExitCode::from(2);
    };
    let ids_only = args.iter().any(|a| a == "--ids-only");
    let gpu = args.iter().any(|a| a == "--gpu");
    let gpu_trait = args.iter().any(|a| a == "--gpu-trait");
    let gpu_argmax = args.iter().any(|a| a == "--gpu-argmax");
    let n_tokens: usize = args
        .get(3)
        .filter(|a| !a.starts_with("--"))
        .map_or(8, |a| a.parse().unwrap_or(8));
    match run(path, prompt, n_tokens, ids_only, gpu, gpu_trait, gpu_argmax) {
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

/// The `k` highest logits as (id, logit, softmax probability over
/// the whole vocabulary), best first. Probabilities are computed in
/// f64 from the forward's finished (softcapped, scaled) logits.
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

fn run(
    path: &str,
    prompt: &str,
    n_tokens: usize,
    ids_only: bool,
    gpu: bool,
    gpu_trait: bool,
    gpu_argmax: bool,
) -> Result<(), String> {
    let t_open = Instant::now();
    let file = gguf::GgufFile::open(Path::new(path)).map_err(|e| e.to_string())?;
    println!("phase=open_file ms={}", t_open.elapsed().as_millis());
    println!("file: {path} bytes={}", file.file_len());

    let mut cfg = itak_loader::load_config(&file).map_err(|e| e.to_string())?;
    // EXPERIMENT ONLY: see `experiment_logit_scale`. Defaults to the
    // file value; shipped behavior is unchanged.
    let file_scale = cfg.logit_scale;
    cfg.logit_scale = experiment_logit_scale(
        std::env::var("ITAK_EXPERIMENT_LOGIT_SCALE").ok().as_deref(),
        file_scale,
    )?;
    if cfg.logit_scale != file_scale {
        println!(
            "experiment: logit_scale overridden file={file_scale} forced={}",
            cfg.logit_scale
        );
    }
    println!(
        "config: layers={} hidden={} vocab={} bos={:?} logit_scale={} logit_softcap={}",
        cfg.block_count,
        cfg.hidden,
        cfg.vocab_size,
        file.tokenizer().map(|t| t.bos_token_id).unwrap_or(None),
        cfg.logit_scale,
        cfg.logit_softcap
    );

    // Tokenize before any weight work: the printed id sequence is
    // the exact input both engines are compared on.
    let info = file.tokenizer().map_err(|e| e.to_string())?;
    let tok_model = file
        .metadata()
        .opt_string("tokenizer.ggml.model")
        .map_err(|e| e.to_string())?
        .map(str::to_string);
    let tokenizer =
        Tokenizer::new(data_from_info(&info, tok_model, None)).map_err(|e| e.to_string())?;
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
    if gpu || gpu_argmax {
        return run_resident(
            &cfg, &weights, &ids, n_tokens, &vocab, &tokenizer, gpu_argmax,
        );
    }
    let t_backend = Instant::now();
    let backend: Box<dyn muse_glimmer::Compute> = if gpu_trait {
        Box::new(wgpu_compute::WgpuCompute::new().map_err(|e| e.to_string())?)
    } else {
        Box::new(cpu_compute::CpuCompute::new())
    };
    println!(
        "phase=backend_open backend={} ms={}",
        if gpu_trait { "wgpu-trait" } else { "cpu" },
        t_backend.elapsed().as_millis()
    );
    let model = GlimmerModel::with_diag(cfg, weights, backend, Arc::new(StderrDiag));
    let mut cache = PositionCache::new();

    let mut generated: Vec<u32> = Vec::new();
    let t_prefill = Instant::now();
    let logits = model.forward(&ids, &mut cache).map_err(|e| e.to_string())?;
    let prefill_ms = t_prefill.elapsed().as_millis();
    let top = top_k(&logits, 5);
    println!("phase=prefill ms={prefill_ms} prompt_tokens={}", ids.len());
    print_top("prefill_top", &top, &vocab);
    print_logit_stats("prefill", &logits, model.config());
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

/// The `--gpu` path: the device-resident session. Same output
/// contract as the model path (phase lines, top-5 per step,
/// per-token wall times, generated ids and text), plus the
/// session build timing and the session's counters at the end
/// (submissions and readbacks per chunk are the numbers the
/// resident path exists to fix).
fn run_resident(
    cfg: &muse_glimmer::GlimmerConfig,
    weights: &muse_glimmer::Weights,
    ids: &[u32],
    n_tokens: usize,
    vocab: &[String],
    tokenizer: &Tokenizer,
    argmax_tail: bool,
) -> Result<(), String> {
    let t_backend = Instant::now();
    let backend = wgpu_compute::WgpuCompute::new().map_err(|e| e.to_string())?;
    println!(
        "phase=backend_open backend={} ms={}",
        if argmax_tail {
            "resident-argmax"
        } else {
            "resident"
        },
        t_backend.elapsed().as_millis()
    );
    let t_session = Instant::now();
    let mut session =
        wgpu_compute::ResidentSession::new(&backend, cfg, weights).map_err(|e| e.to_string())?;
    println!("phase=session_build ms={}", t_session.elapsed().as_millis());

    let mut generated: Vec<u32> = Vec::new();
    let t_prefill = Instant::now();
    let mut next = if argmax_tail {
        let (top, logit) = session.forward_argmax(ids).map_err(|e| e.to_string())?;
        let prefill_ms = t_prefill.elapsed().as_millis();
        println!("phase=prefill ms={prefill_ms} prompt_tokens={}", ids.len());
        println!("prefill_argmax id={top} logit={logit:.6}");
        top
    } else {
        let logits = session.forward(ids).map_err(|e| e.to_string())?;
        let prefill_ms = t_prefill.elapsed().as_millis();
        let top = top_k(&logits, 5);
        println!("phase=prefill ms={prefill_ms} prompt_tokens={}", ids.len());
        print_top("prefill_top", &top, vocab);
        top[0].0
    };
    generated.push(next);
    println!(
        "gen step=1 id={next} tok={:?} source=prefill",
        vocab[next as usize]
    );
    std::io::stdout().flush().ok();

    for step in 2..=n_tokens {
        let t_step = Instant::now();
        if argmax_tail {
            let (top, logit) = session.forward_argmax(&[next]).map_err(|e| e.to_string())?;
            let ms = t_step.elapsed().as_millis();
            next = top;
            generated.push(next);
            println!(
                "gen step={step} id={next} tok={:?} ms={ms} argmax_logit={logit:.6}",
                vocab[next as usize]
            );
        } else {
            let logits = session.forward(&[next]).map_err(|e| e.to_string())?;
            let ms = t_step.elapsed().as_millis();
            let top = top_k(&logits, 5);
            print_top(&format!("step{step}_top"), &top, vocab);
            next = top[0].0;
            generated.push(next);
            println!(
                "gen step={step} id={next} tok={:?} ms={ms}",
                vocab[next as usize]
            );
        }
        std::io::stdout().flush().ok();
    }

    let decoded = tokenizer.decode(&generated).map_err(|e| e.to_string())?;
    println!("generated_ids={generated:?}");
    println!("generated_text={decoded:?}");
    let stats = session.stats();
    println!(
        "phase=session_stats forwards={} chunks={} submissions={} aux_submissions={} \
         readbacks={} argmax_readbacks={} dispatches={} tokens={}",
        stats.forwards,
        stats.chunks,
        stats.submissions,
        stats.aux_submissions,
        stats.readbacks,
        stats.argmax_readbacks,
        stats.dispatches,
        stats.tokens
    );
    if let Some(kb) = peak_rss_kb() {
        println!("peak_rss_kb={kb}");
    }
    Ok(())
}

#[cfg(test)]
mod experiment_tests {
    use super::experiment_logit_scale;

    #[test]
    fn unset_env_keeps_the_file_value() {
        assert_eq!(experiment_logit_scale(None, 0.196_116_13), Ok(0.196_116_13));
    }

    #[test]
    fn set_env_forces_the_scale() {
        assert_eq!(experiment_logit_scale(Some("1.0"), 0.196_116_13), Ok(1.0));
        assert_eq!(experiment_logit_scale(Some("0.5"), 0.196_116_13), Ok(0.5));
    }

    #[test]
    fn bad_env_values_refuse() {
        assert!(experiment_logit_scale(Some("bogus"), 0.196_116_13).is_err());
        assert!(experiment_logit_scale(Some("0"), 0.196_116_13).is_err());
        assert!(experiment_logit_scale(Some("-1.0"), 0.196_116_13).is_err());
        assert!(experiment_logit_scale(Some("inf"), 0.196_116_13).is_err());
    }
}
