//! Registered stub architecture modules: families the arch registry
//! routes today whose full forward passes have not landed yet
//! (ARCHITECTURE.md milestone 5: gemma2, phi3, mistral, olmo2,
//! starcoder2, deepseek2 arrive as arch modules one by one).
//!
//! A stub is not a placeholder load. It proves the seam: dispatch
//! reaches the module registered for the file's own
//! `general.architecture` string, and the module reads the family's
//! own metadata keys (the `<arch>.*` prefix) through the real GGUF
//! parser, failing by name on any key it requires. What it does not
//! do is invent a forward: the registry entry for a stub provides
//! CONFIG and COVERAGE capabilities and no FORWARD, so serve-path
//! engine selection refuses it by capability until the real module
//! replaces the stub under the same string.
//!
//! One shared reader serves the three families registered today
//! because the keys every GGUF writer emits for them are the same
//! shape; each family still gets its own named module function, so
//! when a family's real module lands it replaces exactly one
//! function and one registry entry.

use gguf::GgufFile;

use crate::diag;
use crate::error::Result;

/// The configuration a stub module reads from a file: proof the file
/// routed here and the module could describe it in the family's own
/// keys. Deliberately small: block count, hidden size, context
/// length, and the file's tensor count (the inventory a future
/// coverage check will verify against).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StubConfig {
    /// The architecture string the file declared, echoed so the load
    /// result states which module served it.
    pub architecture: String,
    /// Number of transformer blocks, `<arch>.block_count`.
    pub block_count: u32,
    /// Hidden size, `<arch>.embedding_length`.
    pub hidden: u32,
    /// Context length, `<arch>.context_length`.
    pub context_length: u32,
    /// Tensors present in the file's directory.
    pub tensor_count: usize,
}

/// Read the stub configuration for `architecture` from a parsed
/// file. The keys are the family's own: a file routed here under
/// `gemma2` is read through `gemma2.*` keys, never another family's.
///
/// # Errors
/// The `gguf` crate's named errors, unwrapped in kind: a missing
/// `<arch>.block_count` (or sibling) key is a `MissingMetadata`
/// naming that exact key.
pub fn load(file: &GgufFile, architecture: &str) -> Result<StubConfig> {
    let md = file.metadata();
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let cfg = StubConfig {
        architecture: architecture.to_string(),
        block_count: md.get_u32(&key("block_count"))?,
        hidden: md.get_u32(&key("embedding_length"))?,
        context_length: md.get_u32(&key("context_length"))?,
        tensor_count: file.tensors().len(),
    };
    diag::debug(
        "stub_config_loaded",
        &format!(
            "arch={} layers={} hidden={} context={} tensors={}",
            cfg.architecture, cfg.block_count, cfg.hidden, cfg.context_length, cfg.tensor_count
        ),
    );
    Ok(cfg)
}

/// The registered `gemma2` stub module. Gemma 2's real module (post
/// norms, tied embeddings, sliding-window pattern) replaces this
/// when it lands (TODO T-9.9).
///
/// # Errors
/// See [`load`].
pub fn gemma2(file: &GgufFile) -> Result<StubConfig> {
    load(file, "gemma2")
}

/// The registered `olmo2` stub module. OLMo 2's real module (post
/// norms, QK-norms) replaces this when it lands (TODO T-9.9).
///
/// # Errors
/// See [`load`].
pub fn olmo2(file: &GgufFile) -> Result<StubConfig> {
    load(file, "olmo2")
}

/// The registered `deepseek2` stub module. DeepSeek's real module
/// (MLA tensors and MoE experts read under the `deepseek2.*` prefix)
/// replaces this when it lands (TODO T-9.9).
///
/// # Errors
/// See [`load`].
pub fn deepseek2(file: &GgufFile) -> Result<StubConfig> {
    load(file, "deepseek2")
}
