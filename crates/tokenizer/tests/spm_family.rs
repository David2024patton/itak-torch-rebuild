//! Family selection and SentencePiece behavior at the public API,
//! on a small synthetic llama-style vocabulary (the real-vocabulary
//! proof is `tests/spm_golden.rs`).

use tokenizer::{Error, TokenType, Tokenizer, TokenizerData, TokenizerFamily};

/// A minimal SentencePiece vocabulary in GGUF shape: specials, the
/// bare `▁`, a few pieces, and the byte tokens the fallback needs.
fn spm_data() -> TokenizerData {
    let tokens: Vec<String> = vec![
        "<unk>".into(),  // 0
        "<s>".into(),    // 1
        "</s>".into(),   // 2
        "▁".into(),      // 3
        "a".into(),      // 4
        "b".into(),      // 5
        "▁a".into(),     // 6
        "▁ab".into(),    // 7
        "ab".into(),     // 8
        "<0xC3>".into(), // 9
        "<0xA9>".into(), // 10
    ];
    let types = vec![
        TokenType::Unknown,
        TokenType::Control,
        TokenType::Control,
        TokenType::Normal,
        TokenType::Normal,
        TokenType::Normal,
        TokenType::Normal,
        TokenType::Normal,
        TokenType::Normal,
        TokenType::Byte,
        TokenType::Byte,
    ];
    // Scores: "▁a" beats the "ab" merge, then "▁ab" wins overall.
    let scores = vec![0.0, 0.0, 0.0, -4.0, -1.0, -2.0, -0.5, -0.1, -3.0, 0.0, 0.0];
    assert_eq!(tokens.len(), types.len());
    assert_eq!(tokens.len(), scores.len());
    TokenizerData {
        model: Some("llama".to_string()),
        tokens,
        token_types: types,
        // Present in the file but not parsed for this family: even a
        // malformed entry must not refuse construction.
        merges: vec!["not a merge at all".to_string()],
        scores: Some(scores),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        unknown_token_id: Some(0),
        padding_token_id: None,
    }
}

#[test]
fn model_string_selects_the_family() {
    let tok = Tokenizer::new(spm_data()).expect("spm builds");
    assert_eq!(tok.family(), TokenizerFamily::SentencePiece);

    let mut gpt2 = spm_data();
    gpt2.model = Some("gpt2".to_string());
    // The malformed merge is fine for SPM but the byte-level family
    // validates its merge table, so use a well-formed one there.
    gpt2.merges = vec!["▁ a".to_string()];
    let tok = Tokenizer::new(gpt2).expect("gpt2 builds");
    assert_eq!(tok.family(), TokenizerFamily::ByteLevelBpe);

    let mut absent = spm_data();
    absent.model = None;
    absent.merges = vec!["▁ a".to_string()];
    let tok = Tokenizer::new(absent).expect("absent model builds");
    assert_eq!(tok.family(), TokenizerFamily::ByteLevelBpe);
}

#[test]
fn unknown_model_is_refused_by_name() {
    let mut data = spm_data();
    data.model = Some("bert".to_string());
    let err = Tokenizer::new(data).unwrap_err();
    assert_eq!(err, Error::UnsupportedTokenizerModel("bert".to_string()));
}

#[test]
fn scores_length_must_match_the_vocabulary() {
    let mut data = spm_data();
    data.scores = Some(vec![0.0; 3]);
    let err = Tokenizer::new(data).unwrap_err();
    assert_eq!(
        err,
        Error::ScoresLengthMismatch {
            scores: 3,
            tokens: 11
        }
    );
}

#[test]
fn spm_encode_uses_pieces_and_prefix() {
    let tok = Tokenizer::new(spm_data()).expect("spm builds");
    // No BOS from encode (the caller's job); the space prefix picks
    // the word-initial pieces and the merges chain to "▁ab".
    assert_eq!(tok.encode("ab").unwrap(), vec![7]);
    // A character outside the vocabulary falls back to byte tokens.
    assert_eq!(tok.encode("\u{e9}").unwrap(), vec![3, 9, 10]);
}

#[test]
fn spm_special_tokens_still_match_literally() {
    let tok = Tokenizer::new(spm_data()).expect("spm builds");
    // "</s>" is a control token: matched literally, and the plain
    // segment after it takes the space prefix like any fragment
    // following a special (llama.cpp's is_prev_special rule).
    assert_eq!(tok.encode("a</s>a").unwrap(), vec![6, 2, 6]);
}

#[test]
fn spm_decode_unescapes_and_renders_specials() {
    let tok = Tokenizer::new(spm_data()).expect("spm builds");
    assert_eq!(tok.decode(&[7]).unwrap(), " ab");
    assert_eq!(tok.decode(&[6, 5]).unwrap(), " ab");
    // Byte tokens fold back into the character they split.
    assert_eq!(tok.decode(&[3, 9, 10]).unwrap(), " \u{e9}");
    // Control tokens render literally, as with the byte-level family.
    assert_eq!(tok.decode(&[1, 6]).unwrap(), "<s> a");
}
