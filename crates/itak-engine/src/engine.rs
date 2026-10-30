//! The engine worker: one loaded model, one generation at a time.
//!
//! The model lives on a dedicated worker thread because the CPU
//! compute backend serves exactly one sequence (its KV state is
//! per-sequence and not Sync). Connection threads hand jobs over an
//! mpsc channel and forward the streamed events; a second generate
//! job queues behind the running one. Each job builds a fresh
//! [`GlimmerModel`] from the shared, already-loaded weights (a clone
//! of Arc-backed tensor views, so it is cheap) with a fresh
//! [`CpuCompute`] and a fresh [`PositionCache`], which is the
//! per-sequence isolation the backend contract requires.
//!
//! Every failure inside a job is a named [`GenEvent::Error`], never
//! a panic and never a fabricated token: a bad prompt, a context
//! overflow, and a forward refusal are distinct codes the platform
//! maps to distinct HTTP answers.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cpu_compute::CpuCompute;
use muse_glimmer::{GlimmerConfig, GlimmerModel, PositionCache, StderrDiag, Weights};
use tokenizer::{TokenType, Tokenizer, TokenizerData};

use crate::sampler::Sampler;
use crate::streamer::Streamer;

/// Operator configuration for the engine process. Every value comes
/// from a flag or an environment variable (see main.rs); nothing
/// about a deployment is compiled in.
pub struct EngineConfig {
    /// Path to the GGUF file to load.
    pub model_path: PathBuf,
    /// Advertised model name override; None derives the name from
    /// the file (metadata name, else file stem).
    pub model_name: Option<String>,
}

/// Static facts about the loaded model, shared with connection
/// threads for the hello detail and the debug self-report.
pub struct ModelInfo {
    /// The name requests must address (or leave empty).
    pub name: String,
    /// The file the model was loaded from.
    pub path: String,
    /// The file's own architecture string.
    pub architecture: String,
    /// Transformer layer count.
    pub layers: usize,
    /// Hidden width.
    pub hidden: u32,
    /// Vocabulary size.
    pub vocab: usize,
    /// Context length from the file's metadata.
    pub context_length: u32,
    /// Coverage gate result: tensors the arch module requires.
    pub coverage_expected: usize,
    /// Coverage gate result: tensors found and verified.
    pub coverage_found: usize,
}

/// Live counters for the debug self-report.
#[derive(Default)]
pub struct Stats {
    /// Generations completed (a terminal event was sent).
    pub generations: u64,
    /// Tokens generated across all jobs.
    pub tokens_generated: u64,
    /// Prompt tokens of the most recent job.
    pub last_prompt_tokens: usize,
    /// Generated tokens of the most recent job.
    pub last_generated_tokens: usize,
    /// Cache position reached by the most recent job (prompt plus
    /// generated tokens processed into the KV cache).
    pub last_cache_position: usize,
    /// A generation is running right now.
    pub busy: bool,
}

/// One event on a job's reply channel.
pub enum GenEvent {
    /// A piece of completion text and the token that produced it.
    Token {
        /// Decoded text (may be empty when the token's bytes do not
        /// yet complete a character).
        text: String,
        /// The sampled token id.
        token_id: u32,
    },
    /// The generation ended; no more events follow.
    Done {
        /// "stop" (EOS or stop string) or "length" (max_tokens).
        finish_reason: String,
    },
    /// The generation refused or failed, by name.
    Error {
        /// Protocol error code (see protocol.rs).
        code: String,
        /// Human-readable detail.
        message: String,
    },
}

/// One generation job handed from a connection thread to the worker.
pub struct Job {
    /// Requested model name (already checked against the loaded
    /// model by the connection thread).
    pub prompt: String,
    /// Requested token cap; None for the configured default.
    pub max_tokens: Option<usize>,
    /// Sampling temperature; 0 is greedy.
    pub temperature: f64,
    /// Nucleus mass cap.
    pub top_p: f64,
    /// Stop strings.
    pub stop: Vec<String>,
    /// Sampling seed for this job (ignored when greedy).
    pub seed: u64,
    /// Where events go.
    pub reply: Sender<GenEvent>,
}

/// The loaded model, owned by the worker thread. Passed from
/// [`load`] to [`run_worker`]; its fields stay private because the
/// worker is the only consumer.
pub struct Loaded {
    cfg: GlimmerConfig,
    weights: Weights,
    tokenizer: Tokenizer,
    add_bos: bool,
}

/// Adapts the gguf crate's parsed tokenizer info into the tokenizer
/// crate's input data. Same thin adapter as the loader's conformance
/// harness and generation example; it lives in binary code, not in
/// either library.
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

/// Load the model file end to end: parse, config from the file's own
/// metadata, the coverage gate (any missing tensor fails the load by
/// name), weights, and the tokenizer. Called once at startup; a load
/// failure is fatal to the process, because a model that did not
/// load completely must not serve at all.
pub fn load(config: &EngineConfig) -> Result<(Arc<ModelInfo>, Loaded), String> {
    let path = &config.model_path;
    let file = gguf::GgufFile::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let cfg = itak_loader::load_config(&file).map_err(|e| format!("load config: {e}"))?;
    let report = itak_loader::coverage(&file, &cfg);
    report
        .require_complete()
        .map_err(|e| format!("coverage gate: {e}"))?;
    let coverage_expected = report.expected_count();
    let coverage_found = report.found_count();
    let weights =
        itak_loader::load_weights(&file, &cfg).map_err(|e| format!("load weights: {e}"))?;

    let tinfo = file
        .tokenizer()
        .map_err(|e| format!("tokenizer info: {e}"))?;
    let tok_model = file
        .metadata()
        .opt_string("tokenizer.ggml.model")
        .map_err(|e| format!("tokenizer model key: {e}"))?
        .map(str::to_string);
    let tokenizer = Tokenizer::new(data_from_info(&tinfo, tok_model, None))
        .map_err(|e| format!("build tokenizer: {e}"))?;
    let add_bos = file
        .metadata()
        .get_bool("tokenizer.ggml.add_bos_token")
        .unwrap_or(false);

    let architecture = file
        .metadata()
        .opt_string("general.architecture")
        .map_err(|e| format!("architecture key: {e}"))?
        .unwrap_or(muse_glimmer::ARCHITECTURE)
        .to_string();
    let name = config
        .model_name
        .clone()
        .or_else(|| cfg.name.clone())
        .or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "model".to_string());
    let info = ModelInfo {
        name,
        path: path.display().to_string(),
        architecture,
        layers: cfg.layers.len(),
        hidden: cfg.hidden,
        vocab: cfg.vocab_size as usize,
        context_length: cfg.context_length,
        coverage_expected,
        coverage_found,
    };
    Ok((
        Arc::new(info),
        Loaded {
            cfg,
            weights,
            tokenizer,
            add_bos,
        },
    ))
}

/// The worker loop: receive jobs, run them one at a time, until every
/// sender is gone (the process is shutting down).
pub fn run_worker(
    loaded: Loaded,
    stats: Arc<Mutex<Stats>>,
    default_max_tokens: usize,
    jobs: Receiver<Job>,
) {
    for job in jobs {
        run_job(&loaded, &stats, default_max_tokens, job);
    }
}

fn run_job(loaded: &Loaded, stats: &Arc<Mutex<Stats>>, default_max_tokens: usize, job: Job) {
    let Job {
        prompt,
        max_tokens,
        temperature,
        top_p,
        stop,
        seed,
        reply,
    } = job;
    let fail = |code: &str, message: String| {
        let _ = reply.send(GenEvent::Error {
            code: code.to_string(),
            message,
        });
    };

    // Tokenize: the engine owns the tokenizer, so the prompt the
    // model sees is the prompt the conformance proofs cover. BOS is
    // added exactly when the file's metadata says the family does.
    let body = match loaded.tokenizer.encode(&prompt) {
        Ok(ids) => ids,
        Err(e) => return fail("bad_request", format!("tokenize prompt: {e}")),
    };
    let mut ids: Vec<u32> = Vec::new();
    if loaded.add_bos {
        if let Some(bos) = loaded.tokenizer.bos_id() {
            ids.push(bos);
        }
    }
    ids.extend_from_slice(&body);
    if ids.is_empty() {
        return fail("bad_request", "prompt produced no tokens".to_string());
    }
    let max_tokens = max_tokens.filter(|&n| n > 0).unwrap_or(default_max_tokens);
    let context = loaded.cfg.context_length as usize;
    if ids.len() + max_tokens > context {
        return fail(
            "context_overflow",
            format!(
                "prompt of {} tokens plus max_tokens {} exceeds the context length {}",
                ids.len(),
                max_tokens,
                context
            ),
        );
    }
    crate::log_line(&format!(
        "generate start: prompt_tokens={} max_tokens={} temperature={} top_p={}",
        ids.len(),
        max_tokens,
        temperature,
        top_p
    ));
    crate::log_line(&format!("prompt ids: {ids:?}"));

    {
        let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
        st.busy = true;
        st.last_prompt_tokens = ids.len();
        st.last_generated_tokens = 0;
        st.last_cache_position = ids.len();
    }
    let started = Instant::now();
    let outcome = generate_inner(
        loaded,
        &ids,
        max_tokens,
        temperature,
        top_p,
        stop,
        seed,
        &reply,
    );
    let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
    st.busy = false;
    match outcome {
        JobOutcome::Done {
            finish_reason,
            generated,
            text,
        } => {
            st.generations += 1;
            st.tokens_generated += generated as u64;
            st.last_generated_tokens = generated;
            st.last_cache_position = ids.len() + generated;
            drop(st);
            crate::log_line(&format!(
                "generate done: finish_reason={} generated={} elapsed_ms={} text={:?}",
                finish_reason,
                generated,
                started.elapsed().as_millis(),
                text
            ));
            let _ = reply.send(GenEvent::Done { finish_reason });
        }
        JobOutcome::Failed { code, message } => {
            drop(st);
            crate::log_line(&format!(
                "generate failed: code={} message={}",
                code, message
            ));
            fail(&code, message);
        }
        JobOutcome::Aborted { generated } => {
            st.last_generated_tokens = generated;
            drop(st);
            crate::log_line("generate aborted: client disconnected");
        }
    }
}

enum JobOutcome {
    Done {
        finish_reason: String,
        generated: usize,
        text: String,
    },
    Failed {
        code: String,
        message: String,
    },
    Aborted {
        generated: usize,
    },
}

/// The token loop itself: prefill once, then one forward per sampled
/// token against the position cache (the incremental decode the
/// blueprint requires; there is no full re-forward anywhere here).
#[allow(clippy::too_many_arguments)]
fn generate_inner(
    loaded: &Loaded,
    ids: &[u32],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    stop: Vec<String>,
    seed: u64,
    reply: &Sender<GenEvent>,
) -> JobOutcome {
    let model = GlimmerModel::with_diag(
        loaded.cfg.clone(),
        loaded.weights.clone(),
        Box::new(CpuCompute::new()),
        Arc::new(StderrDiag),
    );
    let mut cache = PositionCache::new();
    let mut sampler = Sampler::new(temperature, top_p, seed);
    let mut streamer = Streamer::new(stop);
    let eos = loaded.tokenizer.eos_id();

    let mut logits = match model.forward(ids, &mut cache) {
        Ok(logits) => logits,
        Err(e) => {
            return JobOutcome::Failed {
                code: "forward_failed".to_string(),
                message: format!("prefill: {e}"),
            }
        }
    };
    let mut generated = 0usize;
    let mut last_token = 0u32;
    loop {
        let token = sampler.pick(&logits);
        if Some(token) == eos {
            if !flush(reply, &mut streamer, last_token) {
                return JobOutcome::Aborted { generated };
            }
            return JobOutcome::Done {
                finish_reason: "stop".to_string(),
                generated,
                text: streamer.text().to_string(),
            };
        }
        generated += 1;
        last_token = token;
        let bytes = match loaded.tokenizer.decode_bytes(&[token]) {
            Ok(bytes) => bytes,
            Err(e) => {
                return JobOutcome::Failed {
                    code: "detokenize_failed".to_string(),
                    message: format!("token {token}: {e}"),
                }
            }
        };
        for chunk in streamer.push(&bytes) {
            if reply
                .send(GenEvent::Token {
                    text: chunk,
                    token_id: token,
                })
                .is_err()
            {
                return JobOutcome::Aborted { generated };
            }
        }
        if streamer.stopped() {
            return JobOutcome::Done {
                finish_reason: "stop".to_string(),
                generated,
                text: streamer.text().to_string(),
            };
        }
        if generated >= max_tokens {
            if !flush(reply, &mut streamer, last_token) {
                return JobOutcome::Aborted { generated };
            }
            return JobOutcome::Done {
                finish_reason: "length".to_string(),
                generated,
                text: streamer.text().to_string(),
            };
        }
        match model.forward(&[token], &mut cache) {
            Ok(next) => logits = next,
            Err(e) => {
                return JobOutcome::Failed {
                    code: "forward_failed".to_string(),
                    message: format!("decode step {generated}: {e}"),
                }
            }
        }
    }
}

/// Send the streamer's held-back tail as final token text,
/// attributed to the last generated token (the bytes are its tail).
/// Returns false when the client has gone away.
fn flush(reply: &Sender<GenEvent>, streamer: &mut Streamer, last_token: u32) -> bool {
    for chunk in streamer.finish() {
        if reply
            .send(GenEvent::Token {
                text: chunk,
                token_id: last_token,
            })
            .is_err()
        {
            return false;
        }
    }
    true
}
