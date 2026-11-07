//! N-gram acceptance measurement over recorded text streams: the
//! expectation-setting harness for the real-file run.
//!
//! The synthetic fixture target (the crate's test target) proves
//! the mechanism and the identity property, but its greedy stream
//! is a random-weight model's stream, not text, so its acceptance
//! rates say little about real files. This harness measures the
//! property that actually decides n-gram acceptance on a real
//! model: how much the emitted stream repeats itself. It replays
//! two recorded streams through the proposer in lockstep, under
//! exactly the session's greedy accounting (propose after
//! `pending`, accept the matching prefix, commit
//! `[pending, accepted...]`, emit one correction token), one
//! stream built from a repeated template block and one from
//! varied prose.
//!
//! Tokenization is word-level (each distinct word gets an id in
//! first-appearance order), which overstates repetition versus
//! the real model's subword stream: function words repeat as
//! whole tokens here. Read the rates as directional, with the
//! repetitive-versus-prose gap as the finding; the real-file
//! measurement on Glimmer stays owed and is named in
//! `bench/ngram-spec-2026-11-07.md`.
//!
//! Usage: `cargo run -p itak-spec --example ngram_acceptance -- [depth]`

use itak_spec::{DraftArm, NgramConfig, NgramProposer, Speculator};

/// A templated status block, repeated verbatim: the shape of
/// logs, reports, and boilerplate generation.
const TEMPLATE_BLOCK: &str = "nightly backup completed for volume data one \
    files copied four thousand ninety two bytes written nine hundred twelve \
    megabytes duration forty one seconds status ok next run scheduled at \
    zero two hundred hours retention policy keeps seven daily snapshots and \
    four weekly snapshots verification checksum matched for every file in \
    the volume no errors were recorded during this run";

/// Varied prose: one pass, no repeated sentences.
const PROSE: &str = "the ferry crossed the harbor before sunrise while \
    gulls circled above the empty docks and a cold wind pushed small waves \
    against the pilings below the terminal windows where clerks stamped \
    tickets for passengers carrying baskets of bread fruit and folded \
    newspapers toward the island market and by the time the sun cleared \
    the lighthouse the deckhands had coiled every rope scrubbed the \
    gangway and logged the crossing in a leather book that smelled of \
    diesel and salt although none of them expected anyone ashore to read \
    a single line of it ever again because the office had switched to \
    electronic records years earlier and kept the book only out of habit \
    and a quiet respect for the captains who had filled its earlier pages \
    with careful handwriting during storms that bent the channel markers \
    and drove the smaller boats back into shelter";

/// Map each distinct word to an id in first-appearance order.
fn tokenize(text: &str) -> Vec<u32> {
    let mut vocab: Vec<&str> = Vec::new();
    text.split_whitespace()
        .map(|word| match vocab.iter().position(|&w| w == word) {
            Some(id) => id as u32,
            None => {
                vocab.push(word);
                (vocab.len() - 1) as u32
            }
        })
        .collect()
}

/// Lockstep greedy simulation, the session's accounting replayed
/// at the proposer level (the same simulation the crate's unit
/// tests pin). Returns (rounds, proposed, accepted).
fn simulate(
    stream: &[u32],
    prompt_len: usize,
    depth: usize,
    vocab: usize,
) -> (usize, usize, usize) {
    let mut prop =
        NgramProposer::new(NgramConfig::default(), vocab).expect("default config is valid");
    prop.commit(&stream[..prompt_len]).expect("commit prompt");
    let mut pending = stream[prompt_len];
    let mut pos = prompt_len + 1;
    let (mut rounds, mut proposed, mut accepted_total) = (0usize, 0usize, 0usize);
    let mut no_draws = || 0.0f64;
    while pos < stream.len() {
        rounds += 1;
        let tokens = prop
            .propose_block(pending, depth, DraftArm::Greedy, &mut no_draws)
            .expect("proposal")
            .tokens;
        proposed += tokens.len();
        let mut accepted = 0;
        while accepted < tokens.len()
            && pos + accepted < stream.len()
            && tokens[accepted] == stream[pos + accepted]
        {
            accepted += 1;
        }
        accepted_total += accepted;
        let mut committed = vec![pending];
        committed.extend_from_slice(&stream[pos..pos + accepted]);
        prop.commit(&committed).expect("commit round");
        if pos + accepted >= stream.len() {
            break;
        }
        pending = stream[pos + accepted];
        pos += accepted + 1;
    }
    (rounds, proposed, accepted_total)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let depth: usize = args.get(1).map_or(6, |a| a.parse().unwrap_or(6));

    let mut repetitive_text = String::new();
    for _ in 0..6 {
        repetitive_text.push_str(TEMPLATE_BLOCK);
        repetitive_text.push(' ');
    }
    let repetitive = tokenize(&repetitive_text);
    let prose = tokenize(PROSE);
    let vocab = (repetitive.iter().chain(prose.iter()).max().expect("tokens") + 1) as usize;

    println!("n-gram lockstep acceptance (word-tokenized streams, depth {depth})");
    for (name, stream) in [("repetitive", &repetitive), ("prose", &prose)] {
        let prompt_len = 24.min(stream.len() / 4);
        let (rounds, proposed, accepted) = simulate(stream, prompt_len, depth, vocab);
        let rate = if proposed == 0 {
            0.0
        } else {
            accepted as f64 / proposed as f64
        };
        let tokens_per_round = if rounds == 0 {
            0.0
        } else {
            (stream.len() - prompt_len) as f64 / rounds as f64
        };
        println!(
            "{name}: stream_tokens={} prompt={prompt_len} rounds={rounds} proposed={proposed} \
             accepted={accepted} acceptance_rate={rate:.4} stream_tokens_per_round={tokens_per_round:.3}",
            stream.len()
        );
    }
}
