//! The engine workers: one loaded model, a configured number of
//! serving slots (TODO T-9.14).
//!
//! Each slot is a worker thread because the CPU compute backend
//! serves exactly one sequence (its KV state is per-sequence and
//! not Sync). Connection threads hand jobs over a bounded mpsc
//! channel and forward the streamed events; jobs beyond the slots
//! queue in the channel up to its configured capacity, and a
//! submission past capacity is refused by name (`engine_busy`)
//! instead of queueing without limit. Each job builds a fresh
//! [`GlimmerModel`] from the shared, already-loaded weights (a clone
//! of Arc-backed tensor views, so it is cheap) with a fresh
//! [`CpuCompute`] and a fresh [`PositionCache`], which is the
//! per-sequence isolation the backend contract requires: slots
//! share the weights and nothing mutable.
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
use muse_glimmer::{Compute, GlimmerConfig, GlimmerModel, PositionCache, StderrDiag, Weights};
use tokenizer::{TokenType, Tokenizer, TokenizerData};

use crate::sampler::{Sampler, SamplerConfig};
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
    /// The model's own chat template from the GGUF metadata
    /// (`tokenizer.chat_template`); None when the file carries
    /// none, which the chat_template op refuses by name.
    pub chat_template: Option<String>,
    /// The BOS token's text ("" when the tokenizer has no BOS), for
    /// the template's `bos_token` variable.
    pub bos_token: String,
    /// The EOS token's text ("" when the tokenizer has no EOS).
    pub eos_token: String,
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
    /// Generations running right now (one per busy slot).
    pub active: usize,
    /// Jobs accepted and waiting for a slot.
    pub queued: usize,
    /// Embedding batches completed.
    pub embeddings: u64,
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

/// One unit of work handed from a connection thread to a worker:
/// a generation, or an embedding batch (TODO T-9.15). Both ride the
/// same bounded queue, so the concurrency contract covers both.
pub enum Work {
    /// A generation job.
    Generate(Job),
    /// An embedding job.
    Embed(EmbedJob),
}

/// The outcome of an embedding batch: one vector per input, in
/// order, or the named failure as (code, message).
pub type EmbedOutcome = Result<Vec<Vec<f32>>, (String, String)>;

/// One embedding job: a batch of texts and the channel the vectors
/// (or the named failure) go back on.
pub struct EmbedJob {
    /// The input texts, in order.
    pub inputs: Vec<String>,
    /// Per-input token cap from the operator's configuration
    /// (--max-embed-tokens); an input tokenizing past it refuses
    /// the batch by name instead of occupying a slot without bound.
    pub max_tokens_per_input: usize,
    /// Where the outcome goes.
    pub reply: Sender<EmbedOutcome>,
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
    /// Sampling seed for this job (ignored when greedy). The
    /// connection thread resolves the request's optional seed
    /// against the clock before the job is queued.
    pub seed: u64,
    /// Top-k candidate cap; None or 0 disables truncation.
    pub top_k: Option<usize>,
    /// Presence penalty (OpenAI form); 0 disables.
    pub presence_penalty: f64,
    /// Frequency penalty (OpenAI form); 0 disables.
    pub frequency_penalty: f64,
    /// Whether this job's full prompt ids and generated text may
    /// be written to the engine log (the operator's --log-content).
    /// Off by default: the log lines carry counts and timings.
    pub log_content: bool,
    /// Where events go.
    pub reply: Sender<GenEvent>,
}

/// The loaded model, shared by the worker threads. Passed from
/// [`load`] to [`run_worker`]; its fields stay private because the
/// workers are the only consumers. Cloning is cheap: the weights
/// are Arc-backed tensor views and the tokenizer is shared behind
/// an Arc, so each slot clones the handle set, not the model.
#[derive(Clone)]
pub struct Loaded {
    cfg: GlimmerConfig,
    weights: Weights,
    tokenizer: Arc<Tokenizer>,
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
    let chat_template = file
        .metadata()
        .opt_string("tokenizer.chat_template")
        .map_err(|e| format!("chat template key: {e}"))?
        .map(str::to_string);
    // The template's special-token variables, as text. A special
    // token that does not decode is reported as empty rather than
    // failing the load: the template op can still serve the
    // template, and a template that needs the text will render
    // visibly wrong instead of silently half-loading the model.
    let token_text = |id: Option<u32>| -> String {
        match id {
            Some(id) => tokenizer
                .decode_bytes(&[id])
                .ok()
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .unwrap_or_default(),
            None => String::new(),
        }
    };
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
        chat_template,
        bos_token: token_text(tokenizer.bos_id()),
        eos_token: token_text(tokenizer.eos_id()),
    };
    Ok((
        Arc::new(info),
        Loaded {
            cfg,
            weights,
            tokenizer: Arc::new(tokenizer),
            add_bos,
        },
    ))
}

/// The worker loop: receive jobs and run them, until every sender
/// is gone (the process is shutting down). One worker thread is one
/// serving slot (TODO T-9.14): `slots` workers share the job queue
/// behind a mutex, and each job builds its own model instance,
/// compute backend, and position cache from the shared weights, so
/// concurrent jobs share nothing mutable. The queue in front of the
/// slots is bounded by the channel capacity the caller chose; a
/// submission past it is refused by the connection thread with the
/// `engine_busy` code, never queued without limit and never run
/// concurrently inside another job's state.
pub fn run_worker(
    loaded: Loaded,
    stats: Arc<Mutex<Stats>>,
    default_max_tokens: usize,
    jobs: Arc<Mutex<Receiver<Work>>>,
) {
    loop {
        let work = {
            let rx = jobs.lock().unwrap_or_else(|p| p.into_inner());
            rx.recv()
        };
        match work {
            Ok(Work::Generate(job)) => run_job(&loaded, &stats, default_max_tokens, job),
            Ok(Work::Embed(job)) => run_embed(&loaded, &stats, job),
            Err(_) => return,
        }
    }
}

/// Run one embedding batch: tokenize each input (with BOS exactly
/// when the file's metadata says the family takes one, the same
/// rule generations follow), look the tokens up in the model's own
/// embedding table through the compute backend's embed step, mean
/// pool the rows, and L2-normalize. The result is a bag-of-embeddings
/// vector: an honest product of the model's table, named as such,
/// not a hidden-state pooling the engine does not do.
fn run_embed(loaded: &Loaded, stats: &Arc<Mutex<Stats>>, job: EmbedJob) {
    {
        let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
        st.queued = st.queued.saturating_sub(1);
    }
    let outcome = embed_batch(loaded, &job.inputs, job.max_tokens_per_input);
    if let Err((code, message)) = &outcome {
        // The batch's refusal is recorded under its protocol code
        // (the per-input cap's bad_request lands here) before the
        // outcome goes back to the connection thread.
        crate::diag::error(code, message);
    }
    if outcome.is_ok() {
        let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
        st.embeddings += 1;
    }
    let _ = job.reply.send(outcome);
}

fn embed_batch(
    loaded: &Loaded,
    inputs: &[String],
    max_tokens_per_input: usize,
) -> Result<Vec<Vec<f32>>, (String, String)> {
    let compute = CpuCompute::new();
    let hidden = loaded.cfg.hidden as usize;
    let mut out = Vec::with_capacity(inputs.len());
    for input in inputs {
        let body = loaded
            .tokenizer
            .encode(input)
            .map_err(|e| ("bad_request".to_string(), format!("tokenize input: {e}")))?;
        let ids = assemble_prompt_ids(&body, loaded.add_bos, loaded.tokenizer.bos_id());
        if ids.is_empty() {
            return Err((
                "bad_request".to_string(),
                "input produced no tokens".to_string(),
            ));
        }
        check_embed_input(ids.len(), max_tokens_per_input)?;
        let rows = compute
            .embed(&loaded.weights.embedding, &ids, &loaded.cfg)
            .map_err(|e| ("forward_failed".to_string(), format!("embed lookup: {e}")))?;
        let pooled = mean_pool(&rows, hidden, ids.len());
        out.push(l2_normalize(&pooled).ok_or_else(|| {
            (
                "forward_failed".to_string(),
                "embedding pooled to a zero vector".to_string(),
            )
        })?);
    }
    Ok(out)
}

/// The per-input token cap refusal for an embedding batch,
/// factored out so its shape is unit-testable without a loaded
/// model. Before the cap existed, one batch could carry any number
/// of inputs of any length onto a shared worker slot and starve
/// generation indefinitely (adjudication, embed path unbounded);
/// the batch count half of the cap is enforced by the connection
/// thread in main.rs, this is the per-input half.
fn check_embed_input(tokens: usize, max_tokens: usize) -> Result<(), (String, String)> {
    if tokens > max_tokens {
        return Err((
            "bad_request".to_string(),
            format!("embed input of {tokens} tokens exceeds the per-input maximum of {max_tokens}"),
        ));
    }
    Ok(())
}

/// Mean of `n` rows of width `hidden`, laid out contiguously.
fn mean_pool(rows: &[f32], hidden: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];
    for row in 0..n {
        for (j, acc) in out.iter_mut().enumerate() {
            *acc += rows[row * hidden + j];
        }
    }
    if n > 0 {
        let scale = 1.0 / n as f32;
        for acc in out.iter_mut() {
            *acc *= scale;
        }
    }
    out
}

/// Scale a vector to unit length; None when it has no length.
fn l2_normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(v.iter().map(|x| x / norm).collect())
}

/// Assemble the id sequence a job runs on: the tokenized body,
/// with the BOS id prepended when the file's metadata asks for
/// one AND the body does not already begin with it. A rendered
/// chat template starts with the bos_token text, which tokenizes
/// to the BOS id itself; prepending again would put two BOS ids
/// at the head of every chat prompt, where llama.cpp's tokenizer
/// drops the duplicate (TODO T-9.16, found live by the T-9.17
/// proof run: the engine logged [200000, 200000, ...] for the
/// Glimmer file's own template).
fn assemble_prompt_ids(body: &[u32], add_bos: bool, bos: Option<u32>) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::with_capacity(body.len() + 1);
    if add_bos {
        if let Some(bos) = bos {
            if body.first() != Some(&bos) {
                ids.push(bos);
            }
        }
    }
    ids.extend_from_slice(body);
    ids
}

fn run_job(loaded: &Loaded, stats: &Arc<Mutex<Stats>>, default_max_tokens: usize, job: Job) {
    let Job {
        prompt,
        max_tokens,
        temperature,
        top_p,
        stop,
        seed,
        top_k,
        presence_penalty,
        frequency_penalty,
        log_content,
        reply,
    } = job;
    // The job leaves the queue the moment a slot takes it, however
    // it ends from here (including the early named failures below).
    {
        let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
        st.queued = st.queued.saturating_sub(1);
    }
    let fail = |code: &str, message: String| {
        // Every named job failure is recorded on the diagnostic
        // channel under its protocol code as it is reported: the
        // tokenize bad_request, the empty-prompt bad_request, and
        // the context_overflow refusal all leave through here.
        crate::diag::error(code, &message);
        let _ = reply.send(GenEvent::Error {
            code: code.to_string(),
            message,
        });
    };

    // Tokenize: the engine owns the tokenizer, so the prompt the
    // model sees is the prompt the conformance proofs cover. BOS is
    // added exactly when the file's metadata says the family does,
    // and never twice: a prompt that already begins with the BOS
    // token (a chat template rendering its bos_token, TODO T-9.16)
    // keeps the one it carries, the same dedupe llama.cpp's
    // tokenizer performs (llama-vocab.cpp: add_bos with the first
    // real token already BOS drops the duplicate).
    let body = match loaded.tokenizer.encode(&prompt) {
        Ok(ids) => ids,
        Err(e) => return fail("bad_request", format!("tokenize prompt: {e}")),
    };
    let ids = assemble_prompt_ids(&body, loaded.add_bos, loaded.tokenizer.bos_id());
    if ids.is_empty() {
        return fail("bad_request", "prompt produced no tokens".to_string());
    }
    let max_tokens = max_tokens.filter(|&n| n > 0).unwrap_or(default_max_tokens);
    let context = loaded.cfg.context_length as usize;
    // checked_add, not addition: a max_tokens near usize::MAX (the
    // configured default is operator input too) must refuse as a
    // context overflow, never wrap the sum under the context length
    // and admit an unbounded generation (adjudication H-1).
    let fits = matches!(ids.len().checked_add(max_tokens), Some(total) if total <= context);
    if !fits {
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
    crate::diag::emit(
        "generate_start",
        &format!(
            "generate start: prompt_tokens={} max_tokens={} temperature={} top_p={}",
            ids.len(),
            max_tokens,
            temperature,
            top_p
        ),
    );
    // The full id list is content, not diagnostics: token ids are
    // reversible to prompt text with the model's own tokenizer, so
    // it is logged only under the operator's --log-content. It is
    // also one unbounded line per job (a max-context prompt is on
    // the order of a megabyte on one line), which is half of how
    // the supervisor's stderr pump wedged (adjudication H-2).
    if log_content {
        crate::diag::emit("generate_prompt_ids", &format!("prompt ids: {ids:?}"));
    }

    {
        let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
        st.active += 1;
        st.last_prompt_tokens = ids.len();
        st.last_generated_tokens = 0;
        st.last_cache_position = ids.len();
    }
    let started = Instant::now();
    let outcome = generate_inner(
        loaded,
        &ids,
        max_tokens,
        SamplerConfig {
            temperature,
            top_p,
            top_k: top_k.unwrap_or(0),
            presence_penalty,
            frequency_penalty,
            seed,
        },
        stop,
        &reply,
    );
    let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
    st.active = st.active.saturating_sub(1);
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
            crate::diag::emit(
                "generate_done",
                &done_log_line(
                    &finish_reason,
                    generated,
                    started.elapsed().as_millis(),
                    &text,
                    log_content,
                ),
            );
            let _ = reply.send(GenEvent::Done { finish_reason });
        }
        JobOutcome::Failed { code, message } => {
            drop(st);
            // The job's failure record is this line (the forward
            // and detokenize refusals land here, named in its
            // fields); the error frame is sent directly rather
            // than through fail(), which would record it twice.
            crate::diag::emit(
                "generate_failed",
                &format!("generate failed: code={} message={}", code, message),
            );
            let _ = reply.send(GenEvent::Error { code, message });
        }
        JobOutcome::Aborted { generated } => {
            st.last_generated_tokens = generated;
            drop(st);
            crate::diag::emit("generate_aborted", "generate aborted: client disconnected");
        }
    }
}

/// The generate-done log line. The completion text is user
/// content, so by default the line carries the finish reason, the
/// count, and the elapsed time only; the full text is included
/// solely under the operator's --log-content, the same gate as the
/// prompt id dump in run_job. The stable prefix and field order
/// are unchanged either way, so log readers see one shape.
fn done_log_line(
    finish_reason: &str,
    generated: usize,
    elapsed_ms: u128,
    text: &str,
    log_content: bool,
) -> String {
    if log_content {
        format!(
            "generate done: finish_reason={finish_reason} generated={generated} elapsed_ms={elapsed_ms} text={text:?}"
        )
    } else {
        format!(
            "generate done: finish_reason={finish_reason} generated={generated} elapsed_ms={elapsed_ms}"
        )
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
fn generate_inner(
    loaded: &Loaded,
    ids: &[u32],
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
    stop: Vec<String>,
    reply: &Sender<GenEvent>,
) -> JobOutcome {
    let model = GlimmerModel::with_diag(
        loaded.cfg.clone(),
        loaded.weights.clone(),
        Box::new(CpuCompute::new()),
        Arc::new(StderrDiag),
    );
    let mut cache = PositionCache::new();
    let mut sampler = Sampler::new(sampler_cfg);
    // The prompt counts toward the penalties (the OpenAI form), so
    // it is observed before the first pick.
    sampler.observe(ids);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_ids_prepend_bos_once_and_only_when_asked() {
        // Plain body: BOS goes on the front.
        assert_eq!(
            assemble_prompt_ids(&[954, 7963], true, Some(200000)),
            vec![200000, 954, 7963]
        );
        // A body already starting with BOS (a rendered chat
        // template) keeps exactly one.
        assert_eq!(
            assemble_prompt_ids(&[200000, 200022, 15651], true, Some(200000)),
            vec![200000, 200022, 15651]
        );
        // The family does not add BOS: the body stands as-is.
        assert_eq!(assemble_prompt_ids(&[954], false, Some(200000)), vec![954]);
        // No BOS id known: nothing to prepend.
        assert_eq!(assemble_prompt_ids(&[954], true, None), vec![954]);
        // Empty body with BOS asked: the lone BOS is the prompt.
        assert_eq!(assemble_prompt_ids(&[], true, Some(200000)), vec![200000]);
    }

    #[test]
    fn done_log_line_carries_content_only_under_the_flag() {
        let plain = done_log_line("stop", 7, 1234, "the completion text", false);
        assert!(plain.contains("finish_reason=stop"), "line: {plain}");
        assert!(plain.contains("generated=7"), "line: {plain}");
        assert!(plain.contains("elapsed_ms=1234"), "line: {plain}");
        assert!(
            !plain.contains("completion text"),
            "content must not reach the default log line: {plain}"
        );
        let full = done_log_line("stop", 7, 1234, "the completion text", true);
        assert!(full.contains("the completion text"), "line: {full}");
    }

    #[test]
    fn embed_input_cap_refuses_by_name() {
        assert!(check_embed_input(8192, 8192).is_ok());
        let (code, message) = check_embed_input(8193, 8192).expect_err("past the cap refuses");
        assert_eq!(code, "bad_request");
        assert!(message.contains("8193"), "message: {message}");
        assert!(message.contains("8192"), "message: {message}");
    }

    #[test]
    fn mean_pool_averages_rows() {
        // Two rows of width 3.
        let rows = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
        assert_eq!(mean_pool(&rows, 3, 2), vec![2.0, 2.0, 2.0]);
        assert_eq!(mean_pool(&rows, 3, 1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn l2_normalize_makes_unit_vectors_and_refuses_zero() {
        let v = l2_normalize(&[3.0, 4.0]).expect("normalizes");
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(l2_normalize(&[0.0, 0.0]).is_none());
    }

    #[test]
    fn pooled_cosine_ranks_the_related_pair_first() {
        // The T-9.15 ranking shape at the pooling level: rows are
        // token embeddings; two inputs sharing a token direction
        // must land closer than an unrelated one.
        let cat = mean_pool(&[1.0, 0.1, 0.9, 0.2], 2, 2); // tokens A, B
        let cats = mean_pool(&[1.0, 0.1, 0.8, 0.3], 2, 2); // tokens A, B'
        let car = mean_pool(&[-0.9, 1.0, -1.0, 0.8], 2, 2); // unrelated
        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let an = l2_normalize(a).expect("a");
            let bn = l2_normalize(b).expect("b");
            an.iter().zip(bn.iter()).map(|(x, y)| x * y).sum()
        };
        assert!(
            cos(&cat, &cats) > cos(&cat, &car),
            "related {} should beat unrelated {}",
            cos(&cat, &cats),
            cos(&cat, &car)
        );
    }
}
