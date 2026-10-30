//! E3 real-file proof: the DFlash draft module end to end against
//! the real files on the server.
//!
//! Usage: `dflash_block <draft.gguf> <target.gguf> [prompt] [block_len]`
//!
//! Phases, each timed and printed as it completes:
//!
//! 1. Census: the draft file's metadata and full tensor inventory,
//!    read with this workspace's `gguf` crate.
//! 2. Config + coverage + weights for the draft through
//!    `itak-loader`'s dflash bridge, coverage gate enforced.
//! 3. Target forward with captured layer inputs (the additive
//!    `forward_capturing` variant) over the prompt, on the CPU
//!    reference backend: this is the conditioning source.
//! 4. Injection of the captured rows into the draft's own cache.
//! 5. One noise block forward through the draft, borrowing the
//!    target's embedding table and output head, reporting
//!    per-position top-1 tokens and checking every logit is finite.
//!
//! This is a correctness runner, not a benchmark: the CPU reference
//! dequantizes as it goes, so the target forward is minutes-class.
//! Timings are reported honestly, per phase.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use cpu_compute::CpuCompute;
use dflash::{DflashCache, DflashModel, TargetTensors};
use muse_glimmer::{GlimmerModel, PositionCache};
use tokenizer::{TokenType, Tokenizer, TokenizerData};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(draft_path), Some(target_path)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: dflash_block <draft.gguf> <target.gguf> [prompt] [block_len]");
        return ExitCode::from(2);
    };
    let prompt = args.get(3).map_or("The capital", String::as_str);
    let block_len: usize = args.get(4).map_or(4, |a| a.parse().unwrap_or(4));
    match run(draft_path, target_path, prompt, block_len) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("FAIL: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn phase(label: &str, t: Instant) {
    println!("[{label}] {:.1}s", t.elapsed().as_secs_f64());
    let _ = std::io::stdout().flush();
}

/// Same thin tokenizer adapter as the other integration runners.
fn data_from_info(
    info: &gguf::TokenizerInfo,
    model: Option<String>,
    unknown: Option<u32>,
) -> TokenizerData {
    TokenizerData {
        model,
        tokens: info.tokens.clone(),
        token_types: info
            .token_types
            .as_ref()
            .map(|types| types.iter().map(|&t| TokenType::from_gguf(t)).collect())
            .unwrap_or_default(),
        merges: info.merges.clone().unwrap_or_default(),
        scores: info.scores.clone(),
        bos_token_id: info.bos_token_id,
        eos_token_id: info.eos_token_id,
        unknown_token_id: unknown,
        padding_token_id: info.padding_token_id,
    }
}

fn run(draft_path: &str, target_path: &str, prompt: &str, block_len: usize) -> Result<(), String> {
    let total = Instant::now();

    // Phase 1: census of the draft file with our gguf crate.
    let t = Instant::now();
    let draft_file = gguf::GgufFile::open(Path::new(draft_path)).map_err(|e| e.to_string())?;
    println!(
        "draft file: {} tensors, {} metadata entries",
        draft_file.tensor_names().count(),
        draft_file.metadata().len()
    );
    for name in draft_file.tensor_names() {
        let info = draft_file.tensor_info(name).map_err(|e| e.to_string())?;
        println!("  {name} {:?} {}", info.dims(), info.type_name());
    }
    phase("census", t);

    // Phase 2: draft config, coverage gate, weights.
    let t = Instant::now();
    let dcfg = itak_loader::load_dflash_config(&draft_file).map_err(|e| e.to_string())?;
    println!(
        "draft config: layers={} hidden={} ffn={} q_heads={} kv_heads={} key={} value={} \
         rope_base={} eps={} block_size={} target_layers={:?} window={} pattern={:?} \
         causal={} mask_token={} vocab={} context={}",
        dcfg.block_count,
        dcfg.hidden,
        dcfg.ffn_hidden,
        dcfg.q_heads,
        dcfg.kv_heads,
        dcfg.key_length,
        dcfg.value_length,
        dcfg.rope_freq_base,
        dcfg.rms_eps,
        dcfg.block_size,
        dcfg.target_layers,
        dcfg.sliding_window,
        dcfg.sliding_pattern,
        dcfg.causal_attention,
        dcfg.mask_token_id,
        dcfg.vocab_size,
        dcfg.context_length
    );
    let report = itak_loader::coverage_dflash(&draft_file, &dcfg);
    println!(
        "coverage: expected={} found={} missing={} unexpected={}",
        report.expected_count(),
        report.found_count(),
        report.missing().len(),
        report.unexpected().len()
    );
    if !report.unexpected().is_empty() {
        println!("  unexpected: {:?}", report.unexpected());
    }
    report.require_complete().map_err(|e| e.to_string())?;
    let dweights =
        itak_loader::load_dflash_weights(&draft_file, &dcfg).map_err(|e| e.to_string())?;
    println!(
        "draft weights: layers={} target_hidden={} conditioning_width={} own_embedding={} own_output={}",
        dweights.layers.len(),
        dweights.target_hidden,
        dweights.conditioning_width(),
        dweights.embedding.is_some(),
        dweights.output.is_some()
    );
    let draft = DflashModel::new(dcfg.clone(), dweights);
    phase("draft load", t);

    // Phase 3: target forward with captured layer inputs.
    let t = Instant::now();
    let target_file = gguf::GgufFile::open(Path::new(target_path)).map_err(|e| e.to_string())?;
    let gcfg = itak_loader::load_config(&target_file).map_err(|e| e.to_string())?;
    let gweights = itak_loader::load_weights(&target_file, &gcfg).map_err(|e| e.to_string())?;
    let info = target_file.tokenizer().map_err(|e| e.to_string())?;
    let tok_model = target_file
        .metadata()
        .opt_string("tokenizer.ggml.model")
        .map_err(|e| e.to_string())?
        .map(str::to_string);
    let tokenizer =
        Tokenizer::new(data_from_info(&info, tok_model, None)).map_err(|e| e.to_string())?;
    let mut prompt_ids = tokenizer.encode(prompt).map_err(|e| e.to_string())?;
    let add_bos = target_file
        .metadata()
        .get_bool("tokenizer.ggml.add_bos_token")
        .unwrap_or(false);
    if add_bos {
        if let Some(bos) = info.bos_token_id {
            prompt_ids.insert(0, bos);
        }
    }
    println!("prompt {prompt:?} -> ids {prompt_ids:?}");
    let target = GlimmerModel::new(gcfg.clone(), gweights, Box::new(CpuCompute::new()));
    let capture_layers: Vec<usize> = dcfg.target_layers.iter().map(|&l| l as usize).collect();
    let mut cache = PositionCache::new();
    let (logits, captures) = target
        .forward_capturing(&prompt_ids, &mut cache, &capture_layers)
        .map_err(|e| e.to_string())?;
    println!(
        "target forward: {} positions, {} captures of [{}][{}]",
        prompt_ids.len(),
        captures.len(),
        prompt_ids.len(),
        gcfg.hidden
    );
    for cap in &captures {
        assert_eq!(cap.rows.len(), prompt_ids.len() * gcfg.hidden as usize);
        println!("  captured layer {} input rows", cap.layer);
    }
    let anchor = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .ok_or("empty target logits")?;
    println!(
        "anchor (target argmax): {anchor} {:?}",
        info.tokens[anchor as usize]
    );
    phase("target forward + capture", t);

    // Phase 4: assemble conditioning features in target_layers
    // order and inject them at the prompt positions.
    let t = Instant::now();
    let hidden = gcfg.hidden as usize;
    let n_pos = prompt_ids.len();
    let width = draft.weights().conditioning_width();
    let mut features = vec![0.0f32; n_pos * width];
    for (k, cap) in captures.iter().enumerate() {
        for p in 0..n_pos {
            features[p * width + k * hidden..p * width + (k + 1) * hidden]
                .copy_from_slice(&cap.rows[p * hidden..(p + 1) * hidden]);
        }
    }
    let positions: Vec<usize> = (0..n_pos).collect();
    let mut dcache = DflashCache::new(&dcfg);
    draft
        .inject(&mut dcache, &features, &positions)
        .map_err(|e| e.to_string())?;
    println!("injected positions: {:?}", dcache.positions());
    phase("inject", t);

    // Phase 5: one noise block through the draft.
    let t = Instant::now();
    let source = itak_loader::DflashTensorSource::new(&target_file);
    let emb =
        dflash::TensorSource::tensor(&source, "token_embd.weight").map_err(|e| e.to_string())?;
    let head = dflash::TensorSource::tensor(&source, "output.weight").map_err(|e| e.to_string())?;
    let target_tensors = TargetTensors {
        embedding: Some(&emb),
        output: Some(&head),
    };
    let block: Vec<u32> = draft
        .noise_block(anchor)
        .into_iter()
        .take(block_len)
        .collect();
    println!("noise block: {block:?} at start position {n_pos}");
    let logits = draft
        .forward_block(&dcache, &block, n_pos, &target_tensors)
        .map_err(|e| e.to_string())?;
    assert_eq!(logits.len(), block.len() * dcfg.vocab_size as usize);
    let vocab = dcfg.vocab_size as usize;
    for (i, row) in logits.chunks_exact(vocab).enumerate() {
        assert!(
            row.iter().all(|z| z.is_finite()),
            "non-finite logit in block row {i}"
        );
        let (top, z) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or("empty logit row")?;
        println!(
            "  block position {}: top-1 = {} {:?} (logit {:.4})",
            n_pos + i,
            top,
            info.tokens[top],
            z
        );
    }
    phase("draft block forward", t);

    println!("PASS: E3 real-file draft block forward complete");
    phase("total", total);
    Ok(())
}
