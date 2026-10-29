//! End-to-end tests over a small synthetic vocabulary: all 256
//! byte-mapped single characters, all 256 `<0xXX>` byte tokens, a
//! handful of merges, and two special tokens. If encode and decode are
//! honest, every string here round-trips exactly.

use tokenizer::{byte_to_char, Error, TokenType, Tokenizer, TokenizerData};

const TH_ID: u32 = 512;
const THE_ID: u32 = 513;
const G_THE_ID: u32 = 514;
const UNK_ID: u32 = 517;
const EOS_ID: u32 = 518;

fn synthetic() -> Tokenizer {
    let mut tokens: Vec<String> = Vec::new();
    let mut types: Vec<TokenType> = Vec::new();
    // Ids 0..=255: the byte-mapped character for each byte (id == byte).
    for b in 0u8..=255 {
        tokens.push(byte_to_char(b).to_string());
        types.push(TokenType::Normal);
    }
    // Ids 256..=511: byte tokens.
    for b in 0u8..=255 {
        tokens.push(format!("<0x{b:02X}>"));
        types.push(TokenType::Byte);
    }
    // Ids 512..: merge products, in merge order.
    let merges = ["t h", "th e", "Ġ the", "h e", "Ġ a"];
    for merged in ["th", "the", "Ġthe", "he", "Ġa"] {
        tokens.push(merged.to_string());
        types.push(TokenType::Normal);
    }
    tokens.push("<unk>".to_string());
    types.push(TokenType::Unknown);
    tokens.push("<|end_of_text|>".to_string());
    types.push(TokenType::Control);

    Tokenizer::new(TokenizerData {
        tokens,
        token_types: types,
        merges: merges.iter().map(ToString::to_string).collect(),
        bos_token_id: Some(EOS_ID),
        eos_token_id: Some(EOS_ID),
        unknown_token_id: Some(UNK_ID),
        padding_token_id: None,
    })
    .expect("synthetic vocabulary is well formed")
}

#[test]
fn round_trip_is_identity() {
    let tok = synthetic();
    let samples = [
        "The capital of France is",
        "Hello, world! Price: $3.50 (cheap).",
        "abc 123 4567",
        "spaces  between   words",
        "trailing spaces  ",
        "one\ntwo\r\nthree",
        "don't stop believin'",
        "a+b=c! x_y-z",
        "café au lait",
        "中文 test 漢字",
        "naïve façade",
        "the the the",
    ];
    for s in samples {
        let ids = tok.encode(s).expect("encode");
        let back = tok.decode(&ids).expect("decode");
        assert_eq!(back, s, "round trip failed for {s:?} (ids {ids:?})");
    }
}

#[test]
fn merges_are_applied_by_rank() {
    let tok = synthetic();
    assert_eq!(tok.encode("the").unwrap(), vec![THE_ID]);
    assert_eq!(tok.encode(" the").unwrap(), vec![G_THE_ID]);
    assert_eq!(tok.encode("th").unwrap(), vec![TH_ID]);
    // The merged tokens decode back to real text, space included.
    assert_eq!(tok.decode(&[G_THE_ID]).unwrap(), " the");
}

#[test]
fn g_dot_becomes_a_real_space() {
    let tok = synthetic();
    // The stored token text for a space is the two-byte char Ġ (U+0120).
    let g_id = tok.token_id("Ġ").expect("Ġ is in the vocab");
    assert_eq!(tok.decode(&[g_id]).unwrap(), " ");
    // Encode produces it and decode removes every trace of it.
    let ids = tok.encode("a b").unwrap();
    assert!(ids.contains(&g_id), "ids {ids:?} should contain Ġ");
    let text = tok.decode(&ids).unwrap();
    assert_eq!(text, "a b");
    assert!(!text.contains('Ġ'));
}

#[test]
fn byte_tokens_reassemble_multibyte_characters() {
    let tok = synthetic();
    let byte_id = |b: u8| tok.token_id(&format!("<0x{b:02X}>")).unwrap();
    // é is U+00E9, UTF-8 bytes C3 A9, carried by two byte tokens.
    assert_eq!(tok.decode(&[byte_id(0xC3), byte_id(0xA9)]).unwrap(), "é");
    // 中 is U+4E2D, UTF-8 bytes E4 B8 AD, carried by three byte tokens.
    assert_eq!(
        tok.decode(&[byte_id(0xE4), byte_id(0xB8), byte_id(0xAD)])
            .unwrap(),
        "中"
    );
    // Byte tokens mix freely with ordinary tokens.
    let a = tok.token_id("a").unwrap();
    assert_eq!(
        tok.decode(&[a, byte_id(0xC3), byte_id(0xA9)]).unwrap(),
        "aé"
    );
    // The same bytes carried by ordinary byte-mapped char tokens also
    // reassemble: 'Ã' is byte C3 and '©' is byte A9 in the alphabet.
    let c3 = tok.token_id("Ã").unwrap();
    let a9 = tok.token_id("©").unwrap();
    assert_eq!(tok.decode(&[c3, a9]).unwrap(), "é");
}

#[test]
fn encode_falls_back_to_byte_tokens() {
    // A vocabulary with no char tokens for bytes >= 0x80: encoding
    // "é" must fall back to its two UTF-8 byte tokens.
    let mut tokens: Vec<String> = Vec::new();
    let mut types: Vec<TokenType> = Vec::new();
    for b in 0u8..0x80 {
        tokens.push(byte_to_char(b).to_string());
        types.push(TokenType::Normal);
    }
    for b in 0u8..=255 {
        tokens.push(format!("<0x{b:02X}>"));
        types.push(TokenType::Byte);
    }
    let tok = Tokenizer::new(TokenizerData {
        tokens,
        token_types: types,
        merges: Vec::new(),
        ..TokenizerData::default()
    })
    .unwrap();
    let ids = tok.encode("é").unwrap();
    assert_eq!(
        ids,
        vec![
            tok.token_id("<0xC3>").unwrap(),
            tok.token_id("<0xA9>").unwrap()
        ]
    );
    assert_eq!(tok.decode(&ids).unwrap(), "é");
}

#[test]
fn special_tokens_match_literally_and_roles_are_exposed() {
    let tok = synthetic();
    assert_eq!(tok.bos_id(), Some(EOS_ID));
    assert_eq!(tok.eos_id(), Some(EOS_ID));
    assert_eq!(tok.unk_id(), Some(UNK_ID));
    assert_eq!(tok.pad_id(), None);

    let ids = tok.encode("hi<|end_of_text|>bye").unwrap();
    assert_eq!(ids.iter().filter(|&&id| id == EOS_ID).count(), 1);
    // The special token is not shredded into BPE pieces.
    assert_eq!(ids.len(), 2 + 1 + 3);
    assert_eq!(tok.decode(&ids).unwrap(), "hi<|end_of_text|>bye");
}

#[test]
fn digits_split_in_runs_of_at_most_three() {
    let tok = synthetic();
    let ids = tok.encode("1234567").unwrap();
    assert_eq!(ids.len(), 7);
    assert_eq!(tok.decode(&ids).unwrap(), "1234567");
}

#[test]
fn malformed_inputs_are_named_errors_not_panics() {
    let tok = synthetic();
    assert_eq!(tok.decode(&[999_999]), Err(Error::InvalidTokenId(999_999)));

    assert_eq!(
        Tokenizer::new(TokenizerData::default()).unwrap_err(),
        Error::EmptyVocab
    );

    let bad_merge = TokenizerData {
        tokens: vec!["a".into()],
        merges: vec!["no space here".into()],
        ..TokenizerData::default()
    };
    assert!(matches!(
        Tokenizer::new(bad_merge),
        Err(Error::InvalidMerge { index: 0, .. })
    ));

    let bad_role = TokenizerData {
        tokens: vec!["a".into()],
        eos_token_id: Some(7),
        ..TokenizerData::default()
    };
    assert!(matches!(
        Tokenizer::new(bad_role),
        Err(Error::SpecialTokenOutOfRange { role: "eos", .. })
    ));
}

#[test]
fn unencodable_symbol_without_unk_is_an_error() {
    let tok = Tokenizer::new(TokenizerData {
        tokens: vec!["a".into()],
        ..TokenizerData::default()
    })
    .unwrap();
    assert!(matches!(tok.encode("b"), Err(Error::UnencodableSymbol(_))));
}
