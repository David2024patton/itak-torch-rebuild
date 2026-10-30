//! itak-engine: the Rust half of the iTaK Torch rebuild.
//!
//! Owns one loaded model and serves the platform's socket protocol
//! (see protocol.rs, the one definition of the wire format): hello,
//! ping, debug, and generate with streamed token events. The model
//! runs on a dedicated worker thread because the CPU compute backend
//! serves one sequence at a time; generations queue.
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
//!
//! A model that fails to load (including the tensor coverage gate)
//! exits the process non-zero with the named error: a partial model
//! never serves.

mod engine;
mod json;
mod protocol;
mod sampler;
mod streamer;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use engine::{EngineConfig, GenEvent, Job, ModelInfo, Stats};
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
    let (job_tx, job_rx) = channel::<Job>();
    let default_max_tokens = config.default_max_tokens;
    let worker_stats = Arc::clone(&stats);
    std::thread::spawn(move || {
        engine::run_worker(loaded, worker_stats, default_max_tokens, job_rx);
    });

    if socket_path.exists() {
        log_line(&format!(
            "removing stale socket file {}",
            socket_path.display()
        ));
        if let Err(e) = std::fs::remove_file(&socket_path) {
            log_line(&format!("cannot remove stale socket: {e}"));
            return 1;
        }
    }
    let listener = match UnixListener::bind(&socket_path) {
        Ok(listener) => listener,
        Err(e) => {
            log_line(&format!("bind {} failed: {e}", socket_path.display()));
            return 1;
        }
    };
    log_line(&format!(
        "serving on {} (protocol {})",
        socket_path.display(),
        protocol::PROTO_VERSION
    ));

    let started = Instant::now();
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let tx = job_tx.clone();
                let info = Arc::clone(&info);
                let stats = Arc::clone(&stats);
                std::thread::spawn(move || handle_conn(stream, tx, info, stats, started));
            }
            Err(e) => log_line(&format!("accept failed: {e}")),
        }
    }
    0
}

struct CliConfig {
    model_path: PathBuf,
    socket_path: PathBuf,
    model_name: Option<String>,
    default_max_tokens: usize,
}

fn parse_args() -> Result<CliConfig, i32> {
    let mut model: Option<String> = std::env::var("ITAK_ENGINE_MODEL").ok();
    let mut socket: Option<String> = std::env::var("ITAK_ENGINE_SOCKET").ok();
    let mut name: Option<String> = std::env::var("ITAK_ENGINE_MODEL_NAME").ok();
    let mut default_max: Option<String> = std::env::var("ITAK_ENGINE_DEFAULT_MAX_TOKENS").ok();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let (slot, label) = match args[i].as_str() {
            "--model" => (&mut model, "--model"),
            "--socket" => (&mut socket, "--socket"),
            "--model-name" => (&mut name, "--model-name"),
            "--default-max-tokens" => (&mut default_max, "--default-max-tokens"),
            "--help" | "-h" => {
                eprintln!(
                    "usage: itak-engine --model PATH [--socket PATH] [--model-name NAME] [--default-max-tokens N]"
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
    let default_max_tokens = match default_max {
        Some(text) => match text.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                eprintln!("itak-engine: --default-max-tokens must be a positive integer");
                return Err(2);
            }
        },
        None => 128,
    };
    Ok(CliConfig {
        model_path: PathBuf::from(model),
        socket_path: PathBuf::from(socket.unwrap_or_else(|| "/tmp/itak-engine.sock".to_string())),
        model_name: name,
        default_max_tokens,
    })
}

fn handle_conn(
    stream: UnixStream,
    jobs: Sender<Job>,
    info: Arc<ModelInfo>,
    stats: Arc<Mutex<Stats>>,
    started: Instant,
) {
    let mut writer = match stream.try_clone() {
        Ok(writer) => writer,
        Err(e) => {
            log_line(&format!("connection clone failed: {e}"));
            return;
        }
    };
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return, // client closed
            Ok(_) => {}
            Err(e) => {
                log_line(&format!("connection read failed: {e}"));
                return;
            }
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
        match op {
            "hello" => {
                let theirs = frame.get("proto").and_then(Value::as_f64);
                if let Some(theirs) = theirs {
                    if theirs != protocol::PROTO_VERSION {
                        let msg = format!(
                            "protocol version {theirs} not spoken here; this engine speaks {}",
                            protocol::PROTO_VERSION
                        );
                        if !send(
                            &mut writer,
                            &protocol::error_frame("protocol_mismatch", &msg),
                        ) {
                            return;
                        }
                        continue;
                    }
                }
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
                let frame = debug_info(&info, &stats, started.elapsed().as_secs());
                if !send(&mut writer, &frame) {
                    return;
                }
            }
            "generate" => {
                if !handle_generate(&frame, &mut writer, &jobs, &info) {
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
    jobs: &Sender<Job>,
    info: &Arc<ModelInfo>,
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
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x2545_f491);
    let job = Job {
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop,
        seed,
        reply: reply_tx,
    };
    if jobs.send(job).is_err() {
        return send(
            writer,
            &protocol::error_frame("engine_died", "the engine worker is gone"),
        );
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

fn stats_busy(stats: &Arc<Mutex<Stats>>) -> bool {
    stats.lock().map(|st| st.busy).unwrap_or(false)
}

/// The debug self-report: what is loaded, how its coverage gate
/// went, and what it has served. This is the engine's half of the
/// platform's /debug view.
fn debug_info(info: &ModelInfo, stats: &Arc<Mutex<Stats>>, uptime_secs: u64) -> Value {
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
        ("busy", Value::Bool(st.busy)),
        ("generations", Value::Num(st.generations as f64)),
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
