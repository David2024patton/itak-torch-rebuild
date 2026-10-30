//! Inventory check against a real GGUF file: parse the Glimmer config
//! from the file's own metadata through the itak-loader adapters, check
//! the file's tensor inventory against the muse-glimmer module's
//! expected names, and settle the one name the skeleton flagged as
//! convention-assumed (`output_norm.weight`) against the real file.
//!
//! Usage: `glimmer_inventory <path-to-gguf> [--full]`
//!
//! Default mode reads metadata and the tensor directory only, plus the
//! one `output_norm.weight` fetch through the TensorSource seam.
//! `--full` additionally loads the entire weight set through
//! [`itak_loader::load_weights`] (the real model is 15.59 GiB; the
//! host needs the RAM for the file image plus the served copies).
//! Exit status is 0 only when the config parses and coverage is
//! complete.

use std::path::Path;
use std::process::ExitCode;

use muse_glimmer::{LayerWeights, TensorSource, TensorView};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: glimmer_inventory <path-to-gguf> [--full]");
        return ExitCode::from(2);
    };
    let full = args.iter().any(|a| a == "--full");
    match run(path, full) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("FAIL: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &str, full: bool) -> Result<(), String> {
    let file = gguf::GgufFile::open(Path::new(path)).map_err(|e| e.to_string())?;
    println!("file: {path}");
    println!("file bytes: {}", file.file_len());
    println!("tensors in file: {}", file.tensors().len());

    let cfg = itak_loader::load_config(&file).map_err(|e| e.to_string())?;
    println!(
        "config: name={:?} layers={} context={} hidden={} ffn={} q_heads={} kv_heads={} key_len={} value_len={} window={} vocab={}",
        cfg.name,
        cfg.block_count,
        cfg.context_length,
        cfg.hidden,
        cfg.ffn_hidden,
        cfg.q_heads,
        cfg.kv_heads,
        cfg.key_length,
        cfg.value_length,
        cfg.sliding_window,
        cfg.vocab_size
    );
    println!(
        "config: rope_base={} rms_eps={} logit_softcap={} logit_scale={} pattern_source={:?} global_layers={} sliding_layers={}",
        cfg.rope_freq_base,
        cfg.rms_eps,
        cfg.logit_softcap,
        cfg.logit_scale,
        cfg.pattern_source,
        cfg.global_layer_count(),
        cfg.sliding_layer_count()
    );
    println!("global layer indices: {:?}", cfg.global_layer_indices());

    // Cross-check: the gguf crate's own generic config reader must
    // agree with the arch module's parse of the same metadata.
    let generic = file.model_config().map_err(|e| e.to_string())?;
    println!(
        "cross-check gguf ModelConfig: arch={} layers={} hidden={} vocab={:?}",
        generic.architecture, generic.layer_count, generic.hidden_size, generic.vocab_size
    );
    if generic.layer_count != cfg.block_count
        || generic.hidden_size != cfg.hidden
        || generic.vocab_size != Some(cfg.vocab_size as usize)
    {
        return Err("arch module config and gguf ModelConfig disagree".to_string());
    }

    let report = itak_loader::coverage(&file, &cfg);
    println!(
        "coverage: expected={} found={} missing={} unexpected={}",
        report.expected_count(),
        report.found_count(),
        report.missing().len(),
        report.unexpected().len()
    );
    for name in report.missing() {
        println!("  MISSING {name}");
    }
    for name in report.unexpected() {
        println!("  UNEXPECTED {name}");
    }

    // The convention-assumed name, settled against the real file,
    // fetched through the TensorSource seam itself.
    let source = itak_loader::GgufTensorSource::new(&file);
    match source.tensor("output_norm.weight") {
        Ok(view) => println!(
            "output_norm.weight: PRESENT dtype={:?} shape={:?} elements={} bytes={}",
            view.dtype,
            view.shape,
            view.elements(),
            view.data.len()
        ),
        Err(e) => println!("output_norm.weight: NOT SERVED: {e}"),
    }

    report.require_complete().map_err(|e| e.to_string())?;

    if full {
        let weights = itak_loader::load_weights(&file, &cfg).map_err(|e| e.to_string())?;
        let mut tensors = 3usize;
        let mut bytes = weights.embedding.data.len()
            + weights.final_norm.data.len()
            + weights.output.data.len();
        for layer in &weights.layers {
            tensors += 14;
            bytes += layer_bytes(layer);
        }
        println!(
            "full load: {} layers, {} tensors, {} bytes served through TensorSource",
            weights.layers.len(),
            tensors,
            bytes
        );
        println!(
            "output head: name={} dtype={:?} shape={:?}",
            weights.output.name, weights.output.dtype, weights.output.shape
        );
        println!(
            "final norm (output_norm.weight): dtype={:?} shape={:?}",
            weights.final_norm.dtype, weights.final_norm.shape
        );
    }

    println!("VERDICT: config loaded from the real file; coverage complete");
    Ok(())
}

fn layer_bytes(layer: &LayerWeights) -> usize {
    let views: [&TensorView; 14] = [
        &layer.attn_norm,
        &layer.attn_q,
        &layer.attn_k,
        &layer.attn_v,
        &layer.attn_q_norm,
        &layer.attn_k_norm,
        &layer.attn_output,
        &layer.attn_gate,
        &layer.post_attention_norm,
        &layer.ffn_norm,
        &layer.ffn_gate,
        &layer.ffn_up,
        &layer.ffn_down,
        &layer.post_ffw_norm,
    ];
    views.iter().map(|v| v.data.len()).sum()
}
