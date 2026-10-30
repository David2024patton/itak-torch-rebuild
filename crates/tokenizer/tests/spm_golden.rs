//! SentencePiece conformance against the TinyLlama golden fixture.
//!
//! The fixture (`golden/fixtures/tinyllama-1.1b-chat-v1.0.Q4_K_M.golden.jsonl`)
//! was produced by llama.cpp itself (`llama-tokenize --ids` for the
//! prompt ids, a greedy `llama-server` run for the completions), so
//! its `prompt_token_ids` are the reference tokenizer's ground truth
//! for the 15 golden prompts on this vocabulary. This test rebuilds
//! the tokenizer from the model's own extracted arrays
//! (`conformance/tinyllama.tokdata`, produced by the gguf crate's
//! `tokenizer_extract` example from the real GGUF on the server) and
//! requires:
//!
//! - `[bos] + encode(prompt)` equals the fixture prompt ids exactly,
//!   for all 15 prompts (the BOS is the runner's job, mirroring
//!   llama.cpp's `add_bos`; see `arch-llama/examples/golden_run.rs`);
//! - `decode` of the prompt ids after the BOS reproduces the prompt
//!   with exactly one leading space (the space prefix the encoder
//!   adds, unescaped back);
//! - `decode` of the fixture completion ids reproduces the fixture
//!   completion text (with the terminal stop token excluded when a
//!   record ended on EOS, the golden runner's rule).

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use tokenizer::{TokenType, Tokenizer, TokenizerData, TokenizerFamily};

const TOKDATA: &str = "conformance/tinyllama.tokdata";
const FIXTURE: &str = "../../golden/fixtures/tinyllama-1.1b-chat-v1.0.Q4_K_M.golden.jsonl";

/// The framed reader for the `.tokdata` format (see the gguf crate's
/// `tokenizer_extract` example for the format definition). A compact
/// copy of the conformance harness's reader: the two consumers stay
/// independent on purpose.
struct Frame {
    inner: BufReader<fs::File>,
}

impl Frame {
    fn open(path: &Path) -> Frame {
        Frame {
            inner: BufReader::new(fs::File::open(path).expect("tokdata opens")),
        }
    }

    fn line(&mut self) -> String {
        let mut s = String::new();
        self.inner.read_line(&mut s).expect("tokdata line");
        s.trim_end_matches(['\n', '\r']).to_string()
    }

    fn field(&mut self, key: &str) -> String {
        let line = self.line();
        line.strip_prefix(key)
            .and_then(|rest| rest.strip_prefix(' '))
            .map(str::to_string)
            .unwrap_or_else(|| panic!("expected '{key} ...' line, found {line:?}"))
    }

    fn framed_string(&mut self) -> String {
        let len: usize = self.line().parse().expect("frame length");
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf).expect("frame bytes");
        let mut nl = [0u8; 1];
        self.inner.read_exact(&mut nl).expect("frame newline");
        assert_eq!(nl[0], b'\n');
        String::from_utf8(buf).expect("frame is UTF-8")
    }
}

fn load_tinyllama() -> (Tokenizer, Option<u32>) {
    let mut frame = Frame::open(Path::new(TOKDATA));
    assert_eq!(frame.line(), "ITAK-TOKDATA-2");
    let model = frame.field("model");
    let _pre = frame.field("pre");
    let parse_id = |v: String| {
        if v == "-" {
            None
        } else {
            Some(v.parse::<u32>().expect("special id"))
        }
    };
    let bos = parse_id(frame.field("bos"));
    let eos = parse_id(frame.field("eos"));
    let unk = parse_id(frame.field("unk"));
    let pad = parse_id(frame.field("pad"));
    let n_tokens: usize = frame.field("tokens").parse().expect("token count");
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(frame.framed_string());
    }
    let n_types: usize = frame.field("token_types").parse().expect("type count");
    let mut token_types = Vec::with_capacity(n_types);
    for _ in 0..n_types {
        let raw: i32 = frame.line().parse().expect("token type");
        token_types.push(TokenType::from_gguf(raw));
    }
    let n_merges: usize = frame.field("merges").parse().expect("merge count");
    let mut merges = Vec::with_capacity(n_merges);
    for _ in 0..n_merges {
        merges.push(frame.framed_string());
    }
    let n_scores: usize = frame.field("scores").parse().expect("score count");
    let mut scores = Vec::with_capacity(n_scores);
    for _ in 0..n_scores {
        scores.push(frame.line().parse::<f32>().expect("score value"));
    }
    let data = TokenizerData {
        model: Some(model),
        tokens,
        token_types,
        merges,
        scores: Some(scores),
        bos_token_id: bos,
        eos_token_id: eos,
        unknown_token_id: unk,
        padding_token_id: pad,
    };
    let tok = Tokenizer::new(data).expect("tinyllama tokenizer builds");
    (tok, bos)
}

/// One golden fixture record, the fields this test checks.
struct Record {
    id: String,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    completion_token_ids: Vec<u32>,
    completion_text: String,
    ended_with_eos: bool,
}

/// Extracts a `"key": "value"` JSON string field. The record layout
/// puts `"prompt"` before `"prompt_token_ids"`, and the quoted key
/// cannot match inside the longer key, so a plain search is exact
/// for this shape (the same approach as the golden runner).
fn json_string(line: &str, key: &str) -> String {
    let pat = format!("\"{key}\"");
    let at = line
        .find(&pat)
        .unwrap_or_else(|| panic!("no {key} in {line:.60}"))
        + pat.len();
    let rest = line[at..]
        .trim_start()
        .strip_prefix(':')
        .unwrap()
        .trim_start();
    parse_json_string(rest)
}

fn parse_json_string(s: &str) -> String {
    let mut chars = s.chars();
    assert_eq!(chars.next(), Some('"'));
    let mut out = String::new();
    loop {
        match chars.next().expect("string terminates") {
            '"' => return out,
            '\\' => match chars.next().expect("escape") {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'u' => {
                    let hex4 = |chars: &mut std::str::Chars| -> u32 {
                        let mut v = 0u32;
                        for _ in 0..4 {
                            v = v * 16 + chars.next().unwrap().to_digit(16).unwrap();
                        }
                        v
                    };
                    let hi = hex4(&mut chars);
                    if (0xd800..0xdc00).contains(&hi) {
                        assert_eq!(chars.next(), Some('\\'));
                        assert_eq!(chars.next(), Some('u'));
                        let lo = hex4(&mut chars);
                        let c = 0x1_0000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                        out.push(char::from_u32(c).unwrap_or('\u{fffd}'));
                    } else {
                        out.push(char::from_u32(hi).unwrap_or('\u{fffd}'));
                    }
                }
                other => panic!("bad escape \\{other}"),
            },
            c => out.push(c),
        }
    }
}

fn json_ids(line: &str, key: &str) -> Vec<u32> {
    let pat = format!("\"{key}\"");
    let at = line.find(&pat).unwrap_or_else(|| panic!("no {key}")) + pat.len();
    let rest = line[at..]
        .trim_start()
        .strip_prefix(':')
        .unwrap()
        .trim_start();
    let rest = rest.strip_prefix('[').unwrap();
    let end = rest.find(']').unwrap();
    rest[..end]
        .split(',')
        .filter(|w| !w.trim().is_empty())
        .map(|w| w.trim().parse::<u32>().expect("token id"))
        .collect()
}

fn load_records() -> Vec<Record> {
    let text = fs::read_to_string(FIXTURE).expect("fixture reads");
    let mut records = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.contains("\"meta\"") {
            continue;
        }
        records.push(Record {
            id: json_string(line, "id"),
            prompt: json_string(line, "prompt"),
            prompt_token_ids: json_ids(line, "prompt_token_ids"),
            completion_token_ids: json_ids(line, "completion_token_ids"),
            completion_text: json_string(line, "completion_text"),
            ended_with_eos: line.contains("\"ended_with_eos\": true"),
        });
    }
    records
}

#[test]
fn tinyllama_prompt_ids_match_the_golden_fixture() {
    let (tok, bos) = load_tinyllama();
    assert_eq!(tok.family(), TokenizerFamily::SentencePiece);
    let bos = bos.expect("tinyllama records a bos");
    let records = load_records();
    assert_eq!(records.len(), 15, "the golden set has 15 prompts");
    for record in &records {
        let mut ids = vec![bos];
        ids.extend(tok.encode(&record.prompt).expect("encode"));
        assert_eq!(
            ids, record.prompt_token_ids,
            "prompt {} tokenization diverges",
            record.id
        );
    }
}

#[test]
fn tinyllama_decode_reproduces_fixture_text() {
    let (tok, _bos) = load_tinyllama();
    for record in load_records() {
        // Prompt: the encoder's added space prefix comes back as one
        // leading space, the documented family behavior.
        let prompt_text = tok
            .decode(&record.prompt_token_ids[1..])
            .expect("prompt decode");
        assert_eq!(prompt_text, format!(" {}", record.prompt), "{}", record.id);
        // Completion: the terminal stop token carries no text in the
        // golden records, so it is excluded when a record ended on
        // EOS (the golden runner's rule).
        let ids = if record.ended_with_eos {
            &record.completion_token_ids[..record.completion_token_ids.len() - 1]
        } else {
            &record.completion_token_ids[..]
        };
        let text = tok.decode(ids).expect("completion decode");
        assert_eq!(text, record.completion_text, "{}", record.id);
    }
}
