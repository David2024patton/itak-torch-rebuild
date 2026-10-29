//! Input data for the tokenizer: the parsed contents of the GGUF
//! `tokenizer.ggml.*` metadata, as plain vectors so this crate stays
//! decoupled from the GGUF parser.

/// The role of a token, mirroring GGUF `tokenizer.ggml.token_type`
/// (llama.cpp's `llama_token_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenType {
    /// No type was recorded for this token.
    #[default]
    Undefined,
    /// An ordinary BPE token whose text is byte-mapped.
    Normal,
    /// The unknown token.
    Unknown,
    /// A control token (BOS, EOS, separators, tool markers). Matched
    /// literally on encode and rendered literally on decode.
    Control,
    /// A user-defined special token. Treated like a control token.
    UserDefined,
    /// Present in the vocabulary but unused by the model.
    Unused,
    /// A byte token: its text has the `<0xXX>` form and it stands for
    /// exactly one raw byte.
    Byte,
}

impl TokenType {
    /// Converts a raw GGUF token-type integer to a [`TokenType`].
    /// Unrecognized values map to [`TokenType::Undefined`].
    pub fn from_gguf(value: i32) -> TokenType {
        match value {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Undefined,
        }
    }

    /// Whether this token is matched literally against the input text
    /// during encoding, instead of going through BPE.
    pub fn is_special(self) -> bool {
        matches!(self, TokenType::Control | TokenType::UserDefined)
    }
}

/// Everything a byte-level BPE tokenizer needs, already parsed.
///
/// `tokens` and `merges` are the GGUF arrays verbatim: token strings
/// are byte-mapped text (a leading space inside a token shows as `Ġ`),
/// and each merge is two token strings separated by one space, in
/// rank order (earlier merges win).
#[derive(Debug, Clone, Default)]
pub struct TokenizerData {
    /// `tokenizer.ggml.tokens`: the vocabulary, indexed by token id.
    pub tokens: Vec<String>,
    /// `tokenizer.ggml.token_type`: parallel to `tokens`. May be empty,
    /// in which case every token is treated as [`TokenType::Undefined`]
    /// and byte tokens are recognized by their `<0xXX>` shape alone.
    pub token_types: Vec<TokenType>,
    /// `tokenizer.ggml.merges`: merge rules in rank order.
    pub merges: Vec<String>,
    /// `tokenizer.ggml.bos_token_id`.
    pub bos_token_id: Option<u32>,
    /// `tokenizer.ggml.eos_token_id`.
    pub eos_token_id: Option<u32>,
    /// `tokenizer.ggml.unknown_token_id`.
    pub unknown_token_id: Option<u32>,
    /// `tokenizer.ggml.padding_token_id`.
    pub padding_token_id: Option<u32>,
}
