//! Session configuration (SPEC-itak-spec Section 6.3). Every operational
//! limit here is configuration with a default, never a compiled-in
//! constant: the depth default is the operating point the llama.cpp sweep
//! measured on 2026-10-30 (`bench/gpu-proof-2026-10-30.md`, depth 6 at
//! 107.39 tok/s against 41.64 plain), and T-9.7 re-measures it on this
//! stack.

/// Which drafting machinery a session runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecMode {
    /// No speculation: the session generates plainly and reports
    /// `spec.disabled` once at creation.
    Off,
    /// DFlash block drafting (the mode this crate implements).
    Dflash,
    /// N-gram prompt-lookup drafting: named in the spec, machinery not in
    /// this crate; selecting it refuses with `ModeUnavailable`.
    Ngram,
    /// Medusa tree drafting: named in the spec, machinery not in this
    /// crate; selecting it refuses with `ModeUnavailable`.
    Medusa,
}

impl SpecMode {
    /// The mode name as it appears in configuration and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            SpecMode::Off => "off",
            SpecMode::Dflash => "dflash",
            SpecMode::Ngram => "ngram",
            SpecMode::Medusa => "medusa",
        }
    }
}

/// What a round does when the target's verify forward refuses
/// (SPEC-itak-spec Section 4, step 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnVerifyError {
    /// Roll back to the round's base and surface the refusal. The default:
    /// a sick target is never papered over.
    Refuse,
    /// Roll back, latch speculation off for the rest of the session with a
    /// `spec.fallback.latched` event, and finish the round as one plain
    /// step. Never silent: the latch is in the stats and the event log.
    LatchOff,
}

/// Everything that tunes a session, validated at creation.
#[derive(Debug, Clone)]
pub struct SpecConfig {
    /// Drafting mode. Default: DFlash.
    pub mode: SpecMode,
    /// Maximum draft tokens proposed per round. Default: 6, the operating
    /// point from the 2026-10-30 sweep. Must be at least 1 and at most the
    /// draft's block size; creation refuses out-of-range depths by name.
    pub depth: usize,
    /// Per-round token bound handed to the KV checkpoint: a round touches
    /// at most this many positions past its base. Must hold `depth + 1`
    /// (the pending token plus the drafts). Default: 17, llama.cpp's own
    /// default speculation envelope.
    pub max_round_tokens: usize,
    /// Target sampling temperature. Zero (the default) is greedy: the
    /// emitted sequence is then token-identical to non-speculative greedy
    /// generation, which is the loop's standing proof obligation.
    pub temperature: f32,
    /// Nucleus (top-p) cutoff applied to the target distribution when
    /// sampling, with the engine sampler's semantics. Values outside
    /// `(0, 1)` disable truncation. Default: 1.0 (disabled).
    pub top_p: f32,
    /// Temperature for the draft's own distributions. `None` (the default)
    /// follows the target temperature, per the spec's default. The draft
    /// distribution never gets nucleus truncation: q enters the correction
    /// ratio raw, exactly as the draft produced it.
    pub draft_temperature: Option<f32>,
    /// Verify-refusal policy. Default: refuse.
    pub on_verify_error: OnVerifyError,
    /// Whether a grammar constraint is active for this session. Grammar
    /// plus speculation refuses at creation (`GrammarConflict`) until a
    /// grammar-aware correction walk exists.
    pub grammar_active: bool,
    /// Stop tokens: emitting one truncates the stream at that token
    /// (inclusive) and ends the session. Default: empty.
    pub eos_tokens: Vec<u32>,
    /// Seed for the session's private generator. Only temperature sampling
    /// consumes draws. Default: 0 (remapped internally, still
    /// reproducible).
    pub seed: u64,
}

impl Default for SpecConfig {
    fn default() -> Self {
        SpecConfig {
            mode: SpecMode::Dflash,
            depth: 6,
            max_round_tokens: 17,
            temperature: 0.0,
            top_p: 1.0,
            draft_temperature: None,
            on_verify_error: OnVerifyError::Refuse,
            grammar_active: false,
            eos_tokens: Vec::new(),
            seed: 0,
        }
    }
}
