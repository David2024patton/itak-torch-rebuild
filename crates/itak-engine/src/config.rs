//! Engine configuration: every operational limit is a config value
//! with a default here, never a constant compiled into the run path
//! (the project's standing rule). The binary's flags and the
//! `ITAK_ENGINE_*` environment variables resolve into this one
//! struct, and [`EngineConfig`] is the only thing the run path reads.

use std::fmt;
use std::path::PathBuf;

/// Default cap on generated tokens when nothing configures it.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 32;

/// Usage text for the binary, printed on `--help` and on any
/// argument error.
pub const USAGE: &str = "\
usage: itak-engine --model <path-to-gguf> --prompt <text> [options]

options:
  --model <path>          GGUF model file (env: ITAK_ENGINE_MODEL)
  --prompt <text>         prompt to complete (env: ITAK_ENGINE_PROMPT)
  --max-tokens <n>        cap on generated tokens (default: 32, env: ITAK_ENGINE_MAX_TOKENS)
  --context-limit <n>     cap on total context, prompt plus generation;
                          defaults to the model's own context_length
                          (env: ITAK_ENGINE_CONTEXT_LIMIT)
  --no-eos-stop           keep generating past the EOS token
  --help                  print this text";

/// The resolved engine configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Prompt text to tokenize and complete.
    pub prompt: String,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Optional cap on the total context (prompt plus generated
    /// tokens). `None` means the model's own `context_length` rules.
    pub context_limit: Option<usize>,
    /// Stop generation when the model emits its EOS token.
    pub stop_on_eos: bool,
}

impl EngineConfig {
    /// A config for `model_path` and `prompt` with every operational
    /// limit at its default.
    pub fn new(model_path: PathBuf, prompt: String) -> Self {
        EngineConfig {
            model_path,
            prompt,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            context_limit: None,
            stop_on_eos: true,
        }
    }

    /// Resolve a config from CLI arguments (excluding the program
    /// name), falling back to the `ITAK_ENGINE_*` environment for any
    /// value a flag does not supply.
    ///
    /// # Errors
    /// [`ConfigError`] naming the offending flag or the missing
    /// required value; nothing is silently defaulted past a typo.
    pub fn from_args(args: &[String]) -> Result<Self, ConfigError> {
        let mut model: Option<String> = None;
        let mut prompt: Option<String> = None;
        let mut max_tokens: Option<usize> = None;
        let mut context_limit: Option<usize> = None;
        let mut stop_on_eos = true;

        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            match arg.as_str() {
                "--help" | "-h" => return Err(ConfigError::HelpRequested),
                "--no-eos-stop" => stop_on_eos = false,
                "--model" | "--prompt" | "--max-tokens" | "--context-limit" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| ConfigError::MissingValue { flag: arg.clone() })?;
                    match arg.as_str() {
                        "--model" => model = Some(value.clone()),
                        "--prompt" => prompt = Some(value.clone()),
                        "--max-tokens" => {
                            max_tokens = Some(parse_limit("--max-tokens", value)?);
                        }
                        _ => {
                            context_limit = Some(parse_limit("--context-limit", value)?);
                        }
                    }
                    i += 1;
                }
                other => {
                    return Err(ConfigError::UnknownFlag {
                        flag: other.to_string(),
                    });
                }
            }
            i += 1;
        }

        let model = model.or_else(|| env_value("ITAK_ENGINE_MODEL")).ok_or(
            ConfigError::MissingRequired {
                flag: "--model",
                env: "ITAK_ENGINE_MODEL",
            },
        )?;
        let prompt = prompt.or_else(|| env_value("ITAK_ENGINE_PROMPT")).ok_or(
            ConfigError::MissingRequired {
                flag: "--prompt",
                env: "ITAK_ENGINE_PROMPT",
            },
        )?;
        let max_new_tokens = match max_tokens {
            Some(n) => n,
            None => match env_value("ITAK_ENGINE_MAX_TOKENS") {
                Some(v) => parse_limit("ITAK_ENGINE_MAX_TOKENS", &v)?,
                None => DEFAULT_MAX_NEW_TOKENS,
            },
        };
        let context_limit = match context_limit {
            Some(n) => Some(n),
            None => match env_value("ITAK_ENGINE_CONTEXT_LIMIT") {
                Some(v) => Some(parse_limit("ITAK_ENGINE_CONTEXT_LIMIT", &v)?),
                None => None,
            },
        };

        Ok(EngineConfig {
            model_path: PathBuf::from(model),
            prompt,
            max_new_tokens,
            context_limit,
            stop_on_eos,
        })
    }
}

/// An environment variable's value when set and non-empty.
fn env_value(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Parse a positive limit, naming the source on failure.
fn parse_limit(source: &str, value: &str) -> Result<usize, ConfigError> {
    match value.parse::<usize>() {
        Ok(0) => Err(ConfigError::InvalidLimit {
            source: source.to_string(),
            value: value.to_string(),
            detail: "limit must be at least 1".to_string(),
        }),
        Ok(n) => Ok(n),
        Err(e) => Err(ConfigError::InvalidLimit {
            source: source.to_string(),
            value: value.to_string(),
            detail: e.to_string(),
        }),
    }
}

/// A configuration refusal, named so the operator sees exactly which
/// value was rejected and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// `--help` was requested; the caller prints [`USAGE`].
    HelpRequested,
    /// A flag that takes a value was the last argument.
    MissingValue {
        /// The flag missing its value.
        flag: String,
    },
    /// An argument the binary does not know.
    UnknownFlag {
        /// The offending argument.
        flag: String,
    },
    /// A required value came from neither flags nor environment.
    MissingRequired {
        /// The flag that supplies it.
        flag: &'static str,
        /// The environment variable that supplies it.
        env: &'static str,
    },
    /// A limit value did not parse or was zero.
    InvalidLimit {
        /// Flag or environment variable the value came from.
        source: String,
        /// The rejected text.
        value: String,
        /// Why it was rejected.
        detail: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::HelpRequested => write!(f, "help requested"),
            ConfigError::MissingValue { flag } => write!(f, "flag {flag} needs a value"),
            ConfigError::UnknownFlag { flag } => write!(f, "unknown argument {flag:?}"),
            ConfigError::MissingRequired { flag, env } => {
                write!(f, "missing required value: pass {flag} or set {env}")
            }
            ConfigError::InvalidLimit {
                source,
                value,
                detail,
            } => write!(f, "invalid limit from {source}: {value:?} ({detail})"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn defaults_apply_when_only_required_flags_given() {
        let cfg = EngineConfig::from_args(&args(&["--model", "m.gguf", "--prompt", "hi"]))
            .expect("required flags suffice");
        assert_eq!(cfg.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(cfg.context_limit, None);
        assert!(cfg.stop_on_eos);
    }

    #[test]
    fn flags_override_defaults() {
        let cfg = EngineConfig::from_args(&args(&[
            "--model",
            "m.gguf",
            "--prompt",
            "hi",
            "--max-tokens",
            "7",
            "--context-limit",
            "99",
            "--no-eos-stop",
        ]))
        .expect("flags parse");
        assert_eq!(cfg.max_new_tokens, 7);
        assert_eq!(cfg.context_limit, Some(99));
        assert!(!cfg.stop_on_eos);
    }

    #[test]
    fn unknown_flag_is_named() {
        let err = EngineConfig::from_args(&args(&["--modle", "m.gguf"])).unwrap_err();
        assert!(matches!(err, ConfigError::UnknownFlag { .. }));
        assert!(err.to_string().contains("--modle"));
    }

    #[test]
    fn missing_model_is_named() {
        let err = EngineConfig::from_args(&args(&["--prompt", "hi"])).unwrap_err();
        assert!(matches!(err, ConfigError::MissingRequired { .. }));
    }

    #[test]
    fn zero_limit_is_refused() {
        let err = EngineConfig::from_args(&args(&[
            "--model",
            "m.gguf",
            "--prompt",
            "hi",
            "--max-tokens",
            "0",
        ]))
        .unwrap_err();
        assert!(matches!(err, ConfigError::InvalidLimit { .. }));
    }
}
