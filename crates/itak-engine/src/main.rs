//! itak-engine binary: load a GGUF model, tokenize a prompt, run
//! greedy generation, print the tokens.
//!
//! Usage and flags live in [`itak_engine::config::USAGE`]. Token
//! output is line-oriented and flushed per line, matching the
//! integration runners: the prompt ids first, then one line per
//! generated token, then the decoded text and the stop reason.
//! Diagnostics go to stderr through the crates' diag channels
//! (`ITAK_ENGINE_DEBUG`, `ITAK_LOADER_DEBUG`, and the model's own
//! `Diag` sink), never mixed into the token stream on stdout.

use std::io::Write;
use std::process::ExitCode;

use itak_engine::config::USAGE;
use itak_engine::{ConfigError, EngineConfig};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match EngineConfig::from_args(&args) {
        Ok(cfg) => cfg,
        Err(ConfigError::HelpRequested) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match itak_engine::run(&cfg) {
        Ok(outcome) => {
            println!("prompt_ids={:?}", outcome.prompt_ids);
            for (i, (id, text)) in outcome
                .generated_ids
                .iter()
                .zip(&outcome.generated_tokens)
                .enumerate()
            {
                println!("token step={} id={id} text={text:?}", i + 1);
            }
            println!("generated_ids={:?}", outcome.generated_ids);
            println!("generated_text={:?}", outcome.text);
            println!("stop_reason={}", outcome.stop_reason);
            std::io::stdout().flush().ok();
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}
