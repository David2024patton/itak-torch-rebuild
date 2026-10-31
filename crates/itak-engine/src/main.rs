//! itak-engine: the Rust half of the iTaK Torch rebuild.
//!
//! Owns one loaded model and serves the platform's socket protocol
//! (see protocol.rs, the one definition of the wire format): hello,
//! ping, debug, and generate with streamed token events. The model
//! runs on a configured number of worker threads (slots) because the
//! CPU compute backend serves one sequence per instance; jobs queue
//! in a bounded channel and submissions past its capacity are
//! refused with the `engine_busy` code (TODO T-9.14).
//!
//! Configuration is flags, each with an environment fallback, and
//! nothing is compiled in:
//!
//! - `--model PATH` / `ITAK_ENGINE_MODEL`: the GGUF file (required).
//! - `--socket PATH` / `ITAK_ENGINE_SOCKET`: the Unix socket to bind
//!   (default `/tmp/itak-engine.sock`).
//! - `--model-name NAME` / `ITAK_ENGINE_MODEL_NAME`: the name
//!   requests must address; default is the file's own metadata name.
//! - `--default-max-tokens N` / `ITAK_ENGINE_DEFAULT_MAX_TOKENS`:
//!   generation cap when a request omits max_tokens (default 128).
//! - `--slots N` / `ITAK_ENGINE_SLOTS`: concurrent generations
//!   (default 1).
//! - `--max-queued N` / `ITAK_ENGINE_MAX_QUEUED`: generations that
//!   may wait for a slot before submissions refuse as engine_busy
//!   (default 8).
//! - `--max-frame-bytes N` / `ITAK_ENGINE_MAX_FRAME_BYTES`: the
//!   largest protocol frame, in bytes, a connection will read
//!   before refusing it as bad_frame and closing (default
//!   4194304).
//! - `--max-embed-inputs N` / `ITAK_ENGINE_MAX_EMBED_INPUTS`: the
//!   largest embed batch, in inputs, one frame may carry (default
//!   256).
//! - `--max-embed-tokens N` / `ITAK_ENGINE_MAX_EMBED_TOKENS`: the
//!   largest single embed input, in tokens (default 8192).
//! - `--log-content` / `ITAK_ENGINE_LOG_CONTENT` (1 or true): also
//!   log the full prompt token ids and generated text of every job.
//!   Default off: job logs carry counts, finish reasons, and
//!   timings only, because prompts and completions are user content
//!   and the supervisor forwards this stderr into the platform log.
//!
//! A model that fails to load (including the tensor coverage gate)
//! exits the process non-zero with the named error: a partial model
//! never serves.

mod engine;
mod json;
mod protocol;
mod sampler;
mod streamer;

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use engine::{EmbedJob, EmbedOutcome, EngineConfig, GenEvent, Job, ModelInfo, Stats, Work};
use json::Value;

/// One timestamped line to stderr: the engine's diagnostic channel
/// until the diag crate lands.
pub fn log_line(message: &str) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    eprintln!("[itak-engine {secs}] {message}");
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let config = match parse_args() {
        Ok(config) => config,
        Err(code) => return code,
    };
    let socket_path = config.socket_path.clone();
    let engine_config = EngineConfig {
        model_path: config.model_path.clone(),
        model_name: config.model_name.clone(),
    };
    log_line(&format!(
        "loading model {} (this takes seconds: parse, coverage gate, weights)",
        config.model_path.display()
    ));
    let load_started = Instant::now();
    let (info, loaded) = match engine::load(&engine_config) {
        Ok(pair) => pair,
        Err(e) => {
            log_line(&format!("model load FAILED: {e}"));
            return 1;
        }
    };
    log_line(&format!(
        "model loaded in {} ms: name={} arch={} layers={} hidden={} vocab={} context={} coverage={}/{}",
        load_started.elapsed().as_millis(),
        info.name,
        info.architecture,
        info.layers,
        info.hidden,
        info.vocab,
        info.context_length,
        info.coverage_found,
        info.coverage_expected
    ));

    let stats = Arc::new(Mutex::new(Stats::default()));
    // The queue capacity is the concurrency contract (TODO T-9.14):
    // `slots` jobs run at once, up to `max_queued` more wait, and a
    // submission past that is refused with the engine_busy code.
    let capacity = config.slots + config.max_queued;
    let (job_tx, job_rx) = sync_channel::<Work>(capacity);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let default_max_tokens = config.default_max_tokens;
    for slot in 0..config.slots {
        let worker_stats = Arc::clone(&stats);
        let worker_jobs = Arc::clone(&job_rx);
        let worker_loaded = loaded.clone();
        std::thread::spawn(move || {
            engine::run_worker(worker_loaded, worker_stats, default_max_tokens, worker_jobs);
        });
        let _ = slot;
    }

    let listener = match bind_socket(&socket_path) {
        Ok(listener) => listener,
        Err(e) => {
            log_line(&e);
            return 1;
        }
    };
    log_line(&format!(
        "serving on {} (protocol {})",
        socket_path.display(),
        protocol::PROTO_VERSION
    ));

    log_line(&format!(
        "concurrency: slots={} max_queued={} (submissions past capacity refuse as engine_busy)",
        config.slots, config.max_queued
    ));
    let started = Instant::now();
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let tx = job_tx.clone();
                let info = Arc::clone(&info);
                let stats = Arc::clone(&stats);
                let limits = Limits {
                    slots: config.slots,
                    max_queued: config.max_queued,
                    max_frame_bytes: config.max_frame_bytes,
                    max_embed_inputs: config.max_embed_inputs,
                    max_embed_tokens: config.max_embed_tokens,
                    log_content: config.log_content,
                };
                std::thread::spawn(move || handle_conn(stream, tx, info, stats, started, limits));
            }
            Err(e) => log_line(&format!("accept failed: {e}")),
        }
    }
    0
}

/// Bind the serving socket.
///
/// A stale socket file from a previous incarnation is removed, but
/// anything else found at the path is never deleted: the path is
/// operator-controlled (--socket / ITAK_ENGINE_SOCKET), and a
/// mistyped path must not erase a real file, so a non-socket is a
/// named startup refusal. After the bind, the socket file is pinned
/// to owner-only permissions (0600): the socket is the engine's
/// control plane, and the file mode is the first half of its access
/// control (the peer credential check in handle_conn is the second).
fn bind_socket(socket_path: &Path) -> Result<UnixListener, String> {
    if socket_path.exists() {
        let is_socket = std::fs::metadata(socket_path)
            .map(|m| m.file_type().is_socket())
            .unwrap_or(false);
        if !is_socket {
            return Err(format!(
                "{} exists and is not a socket; refusing to remove it",
                socket_path.display()
            ));
        }
        log_line(&format!(
            "removing stale socket file {}",
            socket_path.display()
        ));
        std::fs::remove_file(socket_path)
            .map_err(|e| format!("cannot remove stale socket: {e}"))?;
    }
    let listener = UnixListener::bind(socket_path)
        .map_err(|e| format!("bind {} failed: {e}", socket_path.display()))?;
    std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| format!("cannot set socket permissions: {e}"))?;
    Ok(listener)
}

/// The engine's configured bounds, carried to connection threads
/// so refusals can name the state that refused: the concurrency
/// contract (slots and queue depth), the frame size cap, and the
/// embed batch caps.
#[derive(Clone, Copy)]
struct Limits {
    slots: usize,
    max_queued: usize,
    max_frame_bytes: usize,
    max_embed_inputs: usize,
    max_embed_tokens: usize,
    log_content: bool,
}

struct CliConfig {
    model_path: PathBuf,
    socket_path: PathBuf,
    model_name: Option<String>,
    default_max_tokens: usize,
    slots: usize,
    max_queued: usize,
    max_frame_bytes: usize,
    max_embed_inputs: usize,
    max_embed_tokens: usize,
    log_content: bool,
}

fn parse_args() -> Result<CliConfig, i32> {
    let mut model: Option<String> = std::env::var("ITAK_ENGINE_MODEL").ok();
    let mut socket: Option<String> = std::env::var("ITAK_ENGINE_SOCKET").ok();
    let mut name: Option<String> = std::env::var("ITAK_ENGINE_MODEL_NAME").ok();
    let mut default_max: Option<String> = std::env::var("ITAK_ENGINE_DEFAULT_MAX_TOKENS").ok();
    let mut slots: Option<String> = std::env::var("ITAK_ENGINE_SLOTS").ok();
    let mut max_queued: Option<String> = std::env::var("ITAK_ENGINE_MAX_QUEUED").ok();
    let mut max_frame: Option<String> = std::env::var("ITAK_ENGINE_MAX_FRAME_BYTES").ok();
    let mut max_embed_inputs: Option<String> = std::env::var("ITAK_ENGINE_MAX_EMBED_INPUTS").ok();
    let mut max_embed_tokens: Option<String> = std::env::var("ITAK_ENGINE_MAX_EMBED_TOKENS").ok();
    // A boolean flag, not a value flag: the presence of
    // --log-content, or ITAK_ENGINE_LOG_CONTENT=1/true, turns it on.
    let mut log_content = std::env::var("ITAK_ENGINE_LOG_CONTENT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--log-content" {
            log_content = true;
            i += 1;
            continue;
        }
        let (slot, label) = match args[i].as_str() {
            "--model" => (&mut model, "--model"),
            "--socket" => (&mut socket, "--socket"),
            "--model-name" => (&mut name, "--model-name"),
            "--default-max-tokens" => (&mut default_max, "--default-max-tokens"),
            "--slots" => (&mut slots, "--slots"),
            "--max-queued" => (&mut max_queued, "--max-queued"),
            "--max-frame-bytes" => (&mut max_frame, "--max-frame-bytes"),
            "--max-embed-inputs" => (&mut max_embed_inputs, "--max-embed-inputs"),
            "--max-embed-tokens" => (&mut max_embed_tokens, "--max-embed-tokens"),
            "--help" | "-h" => {
                eprintln!(
                    "usage: itak-engine --model PATH [--socket PATH] [--model-name NAME] [--default-max-tokens N] [--slots N] [--max-queued N] [--max-frame-bytes N] [--max-embed-inputs N] [--max-embed-tokens N] [--log-content]"
                );
                return Err(0);
            }
            other => {
                eprintln!("itak-engine: unknown argument {other:?} (try --help)");
                return Err(2);
            }
        };
        i += 1;
        match args.get(i) {
            Some(value) => *slot = Some(value.clone()),
            None => {
                eprintln!("itak-engine: {label} needs a value");
                return Err(2);
            }
        }
        i += 1;
    }
    let model = match model {
        Some(model) if !model.is_empty() => model,
        _ => {
            eprintln!("itak-engine: no model given: pass --model PATH or set ITAK_ENGINE_MODEL");
            return Err(2);
        }
    };
    let parse_count =
        |text: Option<String>, label: &str, default: usize, min: usize| -> Result<usize, i32> {
            match text {
                Some(text) => match text.parse::<usize>() {
                    Ok(n) if n >= min => Ok(n),
                    _ => {
                        eprintln!("itak-engine: {label} must be an integer >= {min}");
                        Err(2)
                    }
                },
                None => Ok(default),
            }
        };
    let default_max_tokens = parse_count(default_max, "--default-max-tokens", 128, 1)?;
    let slots = parse_count(slots, "--slots", 1, 1)?;
    let max_queued = parse_count(max_queued, "--max-queued", 8, 0)?;
    let max_frame_bytes = parse_count(max_frame, "--max-frame-bytes", 4_194_304, 1024)?;
    let max_embed_inputs = parse_count(max_embed_inputs, "--max-embed-inputs", 256, 1)?;
    let max_embed_tokens = parse_count(max_embed_tokens, "--max-embed-tokens", 8192, 1)?;
    Ok(CliConfig {
        model_path: PathBuf::from(model),
        socket_path: PathBuf::from(socket.unwrap_or_else(|| "/tmp/itak-engine.sock".to_string())),
        model_name: name,
        default_max_tokens,
        slots,
        max_queued,
        max_frame_bytes,
        max_embed_inputs,
        max_embed_tokens,
        log_content,
    })
}

/// The uid of the process on the other end of a connected Unix
/// stream, from the kernel (SO_PEERCRED): not a claim the peer
/// makes, so it cannot be spoofed by anything the peer sends.
fn peer_uid(stream: &UnixStream) -> Option<u32> {
    // SAFETY: cred is writable storage of exactly the size passed
    // in len; getsockopt with SO_PEERCRED on a Unix stream socket
    // fills it and reports failure by return code, checked here.
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let rc = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut cred as *mut libc::ucred as *mut libc::c_void,
            &mut len,
        )
    };
    if rc == 0 {
        Some(cred.uid)
    } else {
        None
    }
}

/// Refuse a connection whose peer runs under a different uid than
/// the engine itself. The platform and the engine run as the same
/// user in every supported deployment; a peer under any other uid
/// is not the platform, whatever frames it sends. This is the
/// second half of the socket's access control, behind the 0600
/// file mode from bind_socket.
fn check_peer(stream: &UnixStream) -> Result<(), String> {
    // SAFETY: geteuid takes no arguments and cannot fail.
    let own = unsafe { libc::geteuid() };
    match peer_uid(stream) {
        Some(uid) if uid == own => Ok(()),
        Some(uid) => Err(format!("peer uid {uid} is not the engine's uid {own}")),
        None => Err("cannot read the peer's credentials".to_string()),
    }
}

fn handle_conn(
    stream: UnixStream,
    jobs: SyncSender<Work>,
    info: Arc<ModelInfo>,
    stats: Arc<Mutex<Stats>>,
    started: Instant,
    limits: Limits,
) {
    let mut writer = match stream.try_clone() {
        Ok(writer) => writer,
        Err(e) => {
            log_line(&format!("connection clone failed: {e}"));
            return;
        }
    };
    if let Err(msg) = check_peer(&stream) {
        log_line(&format!("connection refused: {msg}"));
        let _ = send(&mut writer, &protocol::error_frame("peer_refused", &msg));
        return;
    }
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    // The handshake is a gate, not a convention: until a hello with
    // a matching protocol version completes, no other op is served.
    // Before this, hello was one dispatch arm among six and the
    // version was checked only when the client volunteered it, so
    // any local process could skip the handshake entirely.
    let mut greeted = false;
    loop {
        line.clear();
        // The frame cap is a hard bound on memory: read at most one
        // byte past the cap, which is exactly enough to prove the
        // frame is over it. Before this, read_line grew the buffer
        // until the peer sent a newline or the process ran out of
        // memory, whichever came first (adjudication CB).
        let read = reader
            .by_ref()
            .take((limits.max_frame_bytes + 1) as u64)
            .read_line(&mut line);
        match read {
            Ok(0) => return, // client closed
            Ok(_) => {}
            Err(e) => {
                log_line(&format!("connection read failed: {e}"));
                return;
            }
        }
        if line.len() > limits.max_frame_bytes {
            let msg = format!(
                "frame exceeds the maximum frame size of {} bytes",
                limits.max_frame_bytes
            );
            let _ = send(&mut writer, &protocol::error_frame("bad_frame", &msg));
            return;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let frame = match json::parse(trimmed) {
            Ok(frame) => frame,
            Err(e) => {
                if !send(&mut writer, &protocol::error_frame("bad_frame", &e)) {
                    return;
                }
                continue;
            }
        };
        let op = frame.get("op").and_then(Value::as_str).unwrap_or("");
        if op != "hello" && !greeted {
            let msg = format!("first frame on a connection must be hello, not {op:?}");
            let _ = send(
                &mut writer,
                &protocol::error_frame("protocol_mismatch", &msg),
            );
            return;
        }
        match op {
            "hello" => {
                // The version is required, not volunteered: the
                // platform's client always sends it, and a hello
                // without one has not identified what it speaks.
                // A refused handshake closes the connection; there
                // is nothing to retry on the same socket.
                match frame.get("proto").and_then(Value::as_f64) {
                    Some(v) if v == protocol::PROTO_VERSION => {}
                    Some(v) => {
                        let msg = format!(
                            "protocol version {v} not spoken here; this engine speaks {}",
                            protocol::PROTO_VERSION
                        );
                        let _ = send(
                            &mut writer,
                            &protocol::error_frame("protocol_mismatch", &msg),
                        );
                        return;
                    }
                    None => {
                        let _ = send(
                            &mut writer,
                            &protocol::error_frame(
                                "protocol_mismatch",
                                "hello frame carries no proto version",
                            ),
                        );
                        return;
                    }
                }
                greeted = true;
                let detail = format!(
                    "itak-engine {} model={} arch={}",
                    env!("CARGO_PKG_VERSION"),
                    info.name,
                    info.architecture
                );
                if !send(&mut writer, &protocol::hello_ack(&detail)) {
                    return;
                }
            }
            "ping" => {
                let detail = format!("model={} busy={}", info.name, stats_busy(&stats));
                if !send(&mut writer, &protocol::pong(&detail)) {
                    return;
                }
            }
            "debug" => {
                let frame = debug_info(&info, &stats, started.elapsed().as_secs(), limits);
                if !send(&mut writer, &frame) {
                    return;
                }
            }
            "generate" => {
                if !handle_generate(&frame, &mut writer, &jobs, &info, &stats, limits) {
                    return;
                }
            }
            "embed" => {
                if !handle_embed(&frame, &mut writer, &jobs, &info, &stats, limits) {
                    return;
                }
            }
            "chat_template" => {
                let reply = match &info.chat_template {
                    Some(template) => {
                        protocol::chat_template_frame(template, &info.bos_token, &info.eos_token)
                    }
                    None => protocol::error_frame(
                        "no_chat_template",
                        "the model file carries no tokenizer.chat_template",
                    ),
                };
                if !send(&mut writer, &reply) {
                    return;
                }
            }
            other => {
                let msg = format!("unknown op {other:?}");
                if !send(&mut writer, &protocol::error_frame("unknown_op", &msg)) {
                    return;
                }
            }
        }
    }
}

/// Run one generate frame to its terminal frame. Returns false when
/// the connection is dead and the handler should stop.
fn handle_generate(
    frame: &Value,
    writer: &mut UnixStream,
    jobs: &SyncSender<Work>,
    info: &Arc<ModelInfo>,
    stats: &Arc<Mutex<Stats>>,
    limits: Limits,
) -> bool {
    let req = match protocol::parse_generate(frame) {
        Ok(req) => req,
        Err(e) => {
            return send(writer, &protocol::error_frame("bad_request", &e));
        }
    };
    if !req.model.is_empty() && req.model != info.name {
        let msg = format!(
            "this engine serves model {:?}, not {:?}",
            info.name, req.model
        );
        return send(writer, &protocol::error_frame("model_not_found", &msg));
    }
    let (reply_tx, reply_rx) = channel::<GenEvent>();
    // A request seed makes the sampled run reproducible; only a
    // request without one is seeded from the clock (TODO T-9.13).
    let seed = req.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x2545_f491)
    });
    let job = Job {
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop,
        seed,
        top_k: req.top_k,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        log_content: limits.log_content,
        reply: reply_tx,
    };
    // The bounded queue is the concurrency contract: a submission
    // past capacity refuses by name instead of queueing without
    // limit, and the refusal names the state that refused it.
    match jobs.try_send(Work::Generate(job)) {
        Ok(()) => {
            let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
            st.queued += 1;
        }
        Err(TrySendError::Full(_)) => {
            let msg = format!(
                "engine busy: {} slot(s) running and the queue is full (max_queued {})",
                limits.slots, limits.max_queued
            );
            return send(writer, &protocol::error_frame("engine_busy", &msg));
        }
        Err(TrySendError::Disconnected(_)) => {
            return send(
                writer,
                &protocol::error_frame("engine_died", "the engine worker is gone"),
            );
        }
    }
    loop {
        match reply_rx.recv() {
            Ok(GenEvent::Token { text, token_id }) => {
                if !send(writer, &protocol::token_event(&text, token_id, false, None)) {
                    return false;
                }
            }
            Ok(GenEvent::Done { finish_reason }) => {
                return send(
                    writer,
                    &protocol::token_event("", 0, true, Some(&finish_reason)),
                );
            }
            Ok(GenEvent::Error { code, message }) => {
                return send(writer, &protocol::error_frame(&code, &message));
            }
            Err(_) => {
                // The worker dropped the job without a terminal
                // event: it died, and the platform should hear that
                // by name, not as a silent close.
                return send(
                    writer,
                    &protocol::error_frame(
                        "engine_died",
                        "the engine worker stopped mid-generation",
                    ),
                );
            }
        }
    }
}

/// Run one embed frame to its reply frame: queue the batch behind
/// the same bounded contract as generations, wait for the worker's
/// vectors, and answer with the embeddings frame or the named
/// error. Returns false when the connection is dead.
fn handle_embed(
    frame: &Value,
    writer: &mut UnixStream,
    jobs: &SyncSender<Work>,
    info: &Arc<ModelInfo>,
    stats: &Arc<Mutex<Stats>>,
    limits: Limits,
) -> bool {
    let (model, inputs) = match protocol::parse_embed(frame) {
        Ok(pair) => pair,
        Err(e) => {
            return send(writer, &protocol::error_frame("bad_request", &e));
        }
    };
    if !model.is_empty() && model != info.name {
        let msg = format!("this engine serves model {:?}, not {:?}", info.name, model);
        return send(writer, &protocol::error_frame("model_not_found", &msg));
    }
    // The batch cap is enforced here, before the batch can occupy
    // a slot: a frame may be large (up to --max-frame-bytes) and a
    // batch of tiny inputs inside it is still unbounded work
    // without this. The per-input token half rides the job and is
    // enforced by the worker after tokenization.
    if inputs.len() > limits.max_embed_inputs {
        let msg = format!(
            "embed batch of {} inputs exceeds the maximum of {}",
            inputs.len(),
            limits.max_embed_inputs
        );
        return send(writer, &protocol::error_frame("bad_request", &msg));
    }
    let (reply_tx, reply_rx) = channel::<EmbedOutcome>();
    let job = EmbedJob {
        inputs,
        max_tokens_per_input: limits.max_embed_tokens,
        reply: reply_tx,
    };
    match jobs.try_send(Work::Embed(job)) {
        Ok(()) => {
            let mut st = stats.lock().unwrap_or_else(|p| p.into_inner());
            st.queued += 1;
        }
        Err(TrySendError::Full(_)) => {
            let msg = format!(
                "engine busy: {} slot(s) running and the queue is full (max_queued {})",
                limits.slots, limits.max_queued
            );
            return send(writer, &protocol::error_frame("engine_busy", &msg));
        }
        Err(TrySendError::Disconnected(_)) => {
            return send(
                writer,
                &protocol::error_frame("engine_died", "the engine worker is gone"),
            );
        }
    }
    match reply_rx.recv() {
        Ok(Ok(vectors)) => send(writer, &protocol::embeddings_frame(&info.name, &vectors)),
        Ok(Err((code, message))) => send(writer, &protocol::error_frame(&code, &message)),
        Err(_) => send(
            writer,
            &protocol::error_frame("engine_died", "the engine worker stopped mid-embedding"),
        ),
    }
}

fn stats_busy(stats: &Arc<Mutex<Stats>>) -> bool {
    stats.lock().map(|st| st.active > 0).unwrap_or(false)
}

/// The debug self-report: what is loaded, how its coverage gate
/// went, and what it has served. This is the engine's half of the
/// platform's /debug view. The concurrency fields (slots, active,
/// queued, max_queued) are how an operator sees the T-9.14 contract
/// and which jobs hold it.
fn debug_info(
    info: &ModelInfo,
    stats: &Arc<Mutex<Stats>>,
    uptime_secs: u64,
    limits: Limits,
) -> Value {
    let st = stats.lock().unwrap_or_else(|p| p.into_inner());
    json::obj(vec![
        ("op", Value::Str("debug_info".into())),
        ("proto", Value::Num(protocol::PROTO_VERSION)),
        (
            "engine",
            Value::Str(format!("itak-engine {}", env!("CARGO_PKG_VERSION"))),
        ),
        ("model", Value::Str(info.name.clone())),
        ("model_path", Value::Str(info.path.clone())),
        ("architecture", Value::Str(info.architecture.clone())),
        ("layers", Value::Num(info.layers as f64)),
        ("hidden", Value::Num(f64::from(info.hidden))),
        ("vocab", Value::Num(info.vocab as f64)),
        ("context_length", Value::Num(f64::from(info.context_length))),
        (
            "coverage_expected",
            Value::Num(info.coverage_expected as f64),
        ),
        ("coverage_found", Value::Num(info.coverage_found as f64)),
        ("busy", Value::Bool(st.active > 0)),
        (
            "has_chat_template",
            Value::Bool(info.chat_template.is_some()),
        ),
        ("slots", Value::Num(limits.slots as f64)),
        ("active", Value::Num(st.active as f64)),
        ("queued", Value::Num(st.queued as f64)),
        ("max_queued", Value::Num(limits.max_queued as f64)),
        ("generations", Value::Num(st.generations as f64)),
        ("embeddings", Value::Num(st.embeddings as f64)),
        ("tokens_generated", Value::Num(st.tokens_generated as f64)),
        (
            "last_prompt_tokens",
            Value::Num(st.last_prompt_tokens as f64),
        ),
        (
            "last_generated_tokens",
            Value::Num(st.last_generated_tokens as f64),
        ),
        (
            "last_cache_position",
            Value::Num(st.last_cache_position as f64),
        ),
        ("uptime_secs", Value::Num(uptime_secs as f64)),
    ])
}

/// Write one frame as a line and flush (token events must not sit in
/// a buffer; streaming is the point). False means the peer is gone.
fn send(writer: &mut UnixStream, frame: &Value) -> bool {
    let mut line = json::stringify(frame);
    line.push('\n');
    writer.write_all(line.as_bytes()).is_ok() && writer.flush().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn test_info() -> Arc<ModelInfo> {
        Arc::new(ModelInfo {
            name: "fake".to_string(),
            path: "fake.gguf".to_string(),
            architecture: "muse-glimmer".to_string(),
            layers: 1,
            hidden: 8,
            vocab: 16,
            context_length: 128,
            coverage_expected: 1,
            coverage_found: 1,
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
        })
    }

    fn dummy_job() -> Job {
        // Nobody ever runs this job; it only occupies queue space.
        let (reply_tx, _reply_rx) = channel::<GenEvent>();
        Job {
            prompt: "x".to_string(),
            max_tokens: None,
            temperature: 0.0,
            top_p: 0.0,
            stop: Vec::new(),
            seed: 1,
            top_k: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            log_content: false,
            reply: reply_tx,
        }
    }

    fn temp_socket_path(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "itak-engine-test-{tag}-{}-{nanos}.sock",
            std::process::id()
        ))
    }

    #[test]
    fn stale_regular_file_is_never_deleted() {
        // The attack/accident shape: --socket points at a real file.
        // Startup must refuse by name and leave the file untouched.
        let path = temp_socket_path("notasock");
        std::fs::write(&path, b"precious").expect("write file");
        let err = match bind_socket(&path) {
            Ok(_) => panic!("bind must refuse a non-socket path"),
            Err(e) => e,
        };
        assert!(err.contains("not a socket"), "error: {err}");
        assert_eq!(
            std::fs::read(&path).expect("file still there"),
            b"precious",
            "the file must survive the refused startup"
        );
        std::fs::remove_file(&path).expect("cleanup");
    }

    #[test]
    fn bound_socket_is_owner_only() {
        let path = temp_socket_path("mode");
        let listener = bind_socket(&path).expect("bind");
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "socket mode: {mode:o}");
        drop(listener);
        std::fs::remove_file(&path).expect("cleanup");
    }

    #[test]
    fn stale_socket_file_is_removed_and_rebound() {
        let path = temp_socket_path("stale");
        let first = bind_socket(&path).expect("first bind");
        drop(first);
        // The socket file outlives the listener; the next bind must
        // clear it and serve, not fail with AddrInUse.
        let second = bind_socket(&path).expect("second bind clears the stale socket");
        drop(second);
        std::fs::remove_file(&path).expect("cleanup");
    }

    /// Run handle_conn on a thread over a stream pair and hand back
    /// the client end. The job receiver stays alive in the caller so
    /// the channel the connection holds never disconnects mid-test.
    fn serve_conn() -> (
        BufReader<UnixStream>,
        std::sync::mpsc::Receiver<Work>,
        std::thread::JoinHandle<()>,
    ) {
        serve_conn_with_frame_cap(4_194_304)
    }

    fn serve_conn_with_frame_cap(
        max_frame_bytes: usize,
    ) -> (
        BufReader<UnixStream>,
        std::sync::mpsc::Receiver<Work>,
        std::thread::JoinHandle<()>,
    ) {
        serve_conn_with_limits(Limits {
            slots: 1,
            max_queued: 0,
            max_frame_bytes,
            max_embed_inputs: 256,
            max_embed_tokens: 8192,
            log_content: false,
        })
    }

    fn serve_conn_with_limits(
        limits: Limits,
    ) -> (
        BufReader<UnixStream>,
        std::sync::mpsc::Receiver<Work>,
        std::thread::JoinHandle<()>,
    ) {
        let (engine_side, client_side) = UnixStream::pair().expect("stream pair");
        client_side
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .expect("read timeout");
        let (tx, rx) = sync_channel::<Work>(4);
        let stats = Arc::new(Mutex::new(Stats::default()));
        let handle = std::thread::spawn(move || {
            handle_conn(engine_side, tx, test_info(), stats, Instant::now(), limits);
        });
        (BufReader::new(client_side), rx, handle)
    }

    fn write_frame(client: &mut BufReader<UnixStream>, text: &str) {
        let stream = client.get_mut();
        stream.write_all(text.as_bytes()).expect("write frame");
        stream.write_all(b"\n").expect("write newline");
        stream.flush().expect("flush");
    }

    fn read_reply(client: &mut BufReader<UnixStream>) -> String {
        let mut line = String::new();
        let n = client.read_line(&mut line).expect("read reply");
        assert!(n > 0, "expected a reply frame, got EOF");
        line
    }

    fn assert_closed_after_refusal(client: &mut BufReader<UnixStream>) {
        let mut line = String::new();
        let n = client.read_line(&mut line).expect("read after refusal");
        assert_eq!(n, 0, "the connection must close after the refusal");
    }

    #[test]
    fn peer_check_accepts_the_same_uid() {
        // Both ends of a stream pair are this process, so the peer
        // uid is the engine's own: the check passes. A peer under a
        // different uid cannot be produced inside one test process;
        // that half is the kernel's SO_PEERCRED answer, exercised
        // here only in its accepting direction.
        let (a, _b) = UnixStream::pair().expect("stream pair");
        check_peer(&a).expect("same-uid peer is accepted");
        assert_eq!(peer_uid(&a), Some(unsafe { libc::geteuid() }));
    }

    #[test]
    fn op_before_hello_is_refused_and_the_connection_closes() {
        // The adjudication's attack shape: a local process skips the
        // handshake and goes straight for an op.
        let (mut client, _rx, handle) = serve_conn();
        write_frame(&mut client, r#"{"op":"ping"}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"protocol_mismatch\""), "reply: {reply}");
        assert_closed_after_refusal(&mut client);
        drop(client);
        handle.join().expect("conn thread");
    }

    #[test]
    fn hello_without_proto_is_refused_and_the_connection_closes() {
        let (mut client, _rx, handle) = serve_conn();
        write_frame(&mut client, r#"{"op":"hello"}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"protocol_mismatch\""), "reply: {reply}");
        assert!(reply.contains("no proto"), "reply: {reply}");
        assert_closed_after_refusal(&mut client);
        drop(client);
        handle.join().expect("conn thread");
    }

    #[test]
    fn hello_with_wrong_proto_is_refused_and_the_connection_closes() {
        let (mut client, _rx, handle) = serve_conn();
        write_frame(&mut client, r#"{"op":"hello","proto":99}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"protocol_mismatch\""), "reply: {reply}");
        assert_closed_after_refusal(&mut client);
        drop(client);
        handle.join().expect("conn thread");
    }

    #[test]
    fn hello_then_ping_answers() {
        let (mut client, _rx, handle) = serve_conn();
        write_frame(&mut client, r#"{"op":"hello","proto":1,"client":"test"}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"hello_ack\""), "reply: {reply}");
        write_frame(&mut client, r#"{"op":"ping"}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"pong\""), "reply: {reply}");
        drop(client);
        handle.join().expect("conn thread");
    }

    #[test]
    fn oversized_frame_is_refused_and_the_connection_closes() {
        // The adjudication's attack shape: a frame far past the cap.
        // The engine must refuse it by name at the cap, not buffer it.
        let (mut client, _rx, handle) = serve_conn_with_frame_cap(64);
        write_frame(&mut client, r#"{"op":"hello","proto":1}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"hello_ack\""), "reply: {reply}");
        write_frame(&mut client, &"a".repeat(500));
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"bad_frame\""), "reply: {reply}");
        assert_closed_after_refusal(&mut client);
        drop(client);
        handle.join().expect("conn thread");
    }

    #[test]
    fn unterminated_frame_is_refused_at_the_cap() {
        // The OOM shape exactly: no newline ever arrives. The read
        // must stop one byte past the cap and refuse, not grow.
        let (mut client, _rx, handle) = serve_conn_with_frame_cap(64);
        write_frame(&mut client, r#"{"op":"hello","proto":1}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"hello_ack\""), "reply: {reply}");
        let stream = client.get_mut();
        stream
            .write_all(&[b'x'; 500])
            .expect("write unterminated bytes");
        stream.flush().expect("flush");
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"bad_frame\""), "reply: {reply}");
        assert_closed_after_refusal(&mut client);
        drop(client);
        handle.join().expect("conn thread");
    }

    #[test]
    fn wrapping_max_tokens_is_refused_by_name_over_the_wire() {
        // Adjudication H-1's frame, end to end: the refusal is the
        // named bad_request and the connection stays usable.
        let (mut client, _rx, handle) = serve_conn();
        write_frame(&mut client, r#"{"op":"hello","proto":1}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"hello_ack\""), "reply: {reply}");
        write_frame(
            &mut client,
            r#"{"op":"generate","prompt":"x","max_tokens":18446744073709551616}"#,
        );
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"bad_request\""), "reply: {reply}");
        write_frame(&mut client, r#"{"op":"ping"}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"pong\""), "reply: {reply}");
        drop(client);
        handle.join().expect("conn thread");
    }

    #[test]
    fn oversized_embed_batch_is_refused_by_name() {
        // The embed attack shape: one frame carrying a batch far
        // past the cap. The refusal happens before any slot is
        // occupied, and names the cap.
        let (mut client, _rx, handle) = serve_conn_with_limits(Limits {
            slots: 1,
            max_queued: 0,
            max_frame_bytes: 4_194_304,
            max_embed_inputs: 2,
            max_embed_tokens: 8192,
            log_content: false,
        });
        write_frame(&mut client, r#"{"op":"hello","proto":1}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"hello_ack\""), "reply: {reply}");
        write_frame(&mut client, r#"{"op":"embed","input":["a","b","c"]}"#);
        let reply = read_reply(&mut client);
        assert!(reply.contains("\"bad_request\""), "reply: {reply}");
        assert!(reply.contains("maximum of 2"), "reply: {reply}");
        drop(client);
        handle.join().expect("conn thread");
    }

    #[test]
    fn full_queue_refuses_engine_busy_by_name() {
        // Capacity 1 and it is already full: the next submission is
        // refused with the engine_busy code, not queued, not run.
        let (tx, _rx) = sync_channel::<Work>(1);
        tx.try_send(Work::Generate(dummy_job()))
            .expect("fill the queue");
        let (mut a, mut b) = UnixStream::pair().expect("stream pair");
        let frame = crate::json::parse(r#"{"op":"generate","prompt":"hello"}"#).expect("frame");
        let stats = Arc::new(Mutex::new(Stats::default()));
        let ok = handle_generate(
            &frame,
            &mut a,
            &tx,
            &test_info(),
            &stats,
            Limits {
                slots: 1,
                max_queued: 0,
                max_frame_bytes: 4_194_304,
                max_embed_inputs: 256,
                max_embed_tokens: 8192,
                log_content: false,
            },
        );
        assert!(ok, "the connection survives a busy refusal");
        drop(a);
        let mut text = String::new();
        b.read_to_string(&mut text).expect("read reply");
        assert!(text.contains("\"engine_busy\""), "reply: {text}");
        // The refused job was not counted as queued.
        let st = stats.lock().expect("stats");
        assert_eq!(st.queued, 0);
    }
}
