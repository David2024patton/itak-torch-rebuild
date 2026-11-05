//! Prefill profile harness: where does Glimmer prefill time
//! actually go, per kernel, on the resident GPU path?
//!
//! Usage: `prefill_profile <path-to-gguf> <prompt>`
//!
//! Loads the real file exactly like `glimmer_generate`
//! (gguf parse, `itak-loader` config and weights behind the
//! coverage gate, the same tokenizer adapter), builds the
//! device-resident session in `wgpu-compute`, then feeds the
//! prompt through `ResidentSession::forward` in slices of
//! `PREFILL_PROFILE_SLICE` tokens (default 64, the session's
//! chunk cap), draining the diag timestamp recorder after
//! every slice so no dispatch window is ever lost to the
//! recorder's 2,048-window cap.
//!
//! With `WGPU_COMPUTE_DIAG=timing` set, the run is timed per
//! dispatch: at the end the per-kernel summary table is
//! printed and written to the path in `WGPU_COMPUTE_DIAG_OUT`
//! (the same env the diag smoke collector uses), and the raw
//! per-dispatch records land beside it as
//! `<out>.records.csv` for shape-level analysis. With the
//! diag env unset the recorder does not exist, the encode
//! path is the shipped byte-identical off path, and the run
//! is the timing-off control: same binary, same slices, only
//! the `phase=prefill` wall time is comparable.
//!
//! Wall-time caveat, stated once here and in the bench note:
//! in timing mode each slice boundary pays a drain (a resolve
//! submit plus a blocking map), which serializes the pipeline
//! between slices. The per-kernel GPU durations are measured
//! on the device and are unaffected; the timing-mode wall
//! `phase=prefill` is therefore an upper bound, and the
//! off-mode wall is the prefill wall of record.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tokenizer::{TokenType, Tokenizer, TokenizerData};
use wgpu_compute::timestamps::{render_summary, DIAG_OUT_ENV};

/// Adapts the gguf crate's parsed tokenizer info into the tokenizer
/// crate's input data. Same thin adapter as `glimmer_generate`
/// (TODO T-2.3); it lives in harness code, not in either library.
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

/// Slice-size resolution: the env override applies only inside
/// 1..=1024; anything else falls back to 64, the session's chunk
/// cap and the bench protocol's row count.
fn resolve_slice_rows(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse().ok())
        .filter(|&v| (1..=1024).contains(&v))
        .unwrap_or(64)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(path), Some(prompt)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: prefill_profile <path-to-gguf> <prompt>");
        return ExitCode::from(2);
    };
    match run(path, prompt) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("FAIL: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &str, prompt: &str) -> Result<(), String> {
    let slice_rows = resolve_slice_rows(std::env::var("PREFILL_PROFILE_SLICE").ok().as_deref());

    let file = gguf::GgufFile::open(Path::new(path)).map_err(|e| e.to_string())?;
    println!("file: {path} bytes={}", file.file_len());
    let cfg = itak_loader::load_config(&file).map_err(|e| e.to_string())?;
    println!(
        "config: layers={} hidden={} vocab={}",
        cfg.block_count, cfg.hidden, cfg.vocab_size
    );

    // Tokenize before any weight work, exactly like
    // glimmer_generate, so the prompt id sequence is the bench
    // protocol's own.
    let info = file.tokenizer().map_err(|e| e.to_string())?;
    let tok_model = file
        .metadata()
        .opt_string("tokenizer.ggml.model")
        .map_err(|e| e.to_string())?
        .map(str::to_string);
    let tokenizer =
        Tokenizer::new(data_from_info(&info, tok_model, None)).map_err(|e| e.to_string())?;
    let body = tokenizer.encode(prompt).map_err(|e| e.to_string())?;
    let add_bos = file
        .metadata()
        .get_bool("tokenizer.ggml.add_bos_token")
        .unwrap_or(false);
    let mut ids: Vec<u32> = Vec::new();
    if add_bos {
        if let Some(bos) = info.bos_token_id {
            ids.push(bos);
        }
    }
    ids.extend_from_slice(&body);
    println!("prompt_tokens={} slice_rows={slice_rows}", ids.len());
    std::io::stdout().flush().ok();

    let report = itak_loader::coverage(&file, &cfg);
    report.require_complete().map_err(|e| e.to_string())?;
    let t_load = Instant::now();
    let weights = itak_loader::load_weights(&file, &cfg).map_err(|e| e.to_string())?;
    println!("phase=load_weights ms={}", t_load.elapsed().as_millis());

    let t_backend = Instant::now();
    let backend = wgpu_compute::WgpuCompute::new().map_err(|e| e.to_string())?;
    println!(
        "phase=backend_open backend=resident ms={}",
        t_backend.elapsed().as_millis()
    );
    let ctx = backend.context();
    let recorder = ctx.diag_recorder().cloned();
    let timing = recorder.is_some();
    println!(
        "diag: mode={} recorder={}",
        if timing { "timing" } else { "off" },
        if timing { "present" } else { "absent" }
    );
    if std::env::var("WGPU_COMPUTE_DIAG").is_ok_and(|v| v == "timing") && !timing {
        return Err(
            "WGPU_COMPUTE_DIAG=timing requested but no diag recorder exists \
             (device granted no timestamp features); refusing to profile blind"
                .to_string(),
        );
    }

    let t_session = Instant::now();
    let mut session =
        wgpu_compute::ResidentSession::new(&backend, &cfg, &weights).map_err(|e| e.to_string())?;
    println!("phase=session_build ms={}", t_session.elapsed().as_millis());
    std::io::stdout().flush().ok();

    let t_prefill = Instant::now();
    let mut last_logits: Vec<f32> = Vec::new();
    let mut drained_total = 0usize;
    for (i, slice) in ids.chunks(slice_rows).enumerate() {
        let t_slice = Instant::now();
        last_logits = session.forward(slice).map_err(|e| e.to_string())?;
        let slice_ms = t_slice.elapsed().as_millis();
        let mut drained = 0usize;
        if let Some(rec) = &recorder {
            drained = rec
                .drain(&ctx.device, &ctx.queue)
                .map_err(|e| e.to_string())?;
            drained_total += drained;
        }
        println!(
            "slice={} tokens={} ms={} drained={}",
            i,
            slice.len(),
            slice_ms,
            drained
        );
        std::io::stdout().flush().ok();
    }
    let prefill_ms = t_prefill.elapsed().as_millis();
    println!("phase=prefill ms={prefill_ms} prompt_tokens={}", ids.len());

    // Sanity anchor only (this harness is a profiler, not a
    // correctness gate): the winning first token and its logit.
    if let Some((pos, _)) = last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    {
        println!("prefill_argmax id={pos} logit={:.6}", last_logits[pos]);
    }

    let stats = session.stats();
    println!(
        "phase=session_stats forwards={} chunks={} submissions={} aux_submissions={} \
         readbacks={} argmax_readbacks={} dispatches={} tokens={}",
        stats.forwards,
        stats.chunks,
        stats.submissions,
        stats.aux_submissions,
        stats.readbacks,
        stats.argmax_readbacks,
        stats.dispatches,
        stats.tokens
    );

    if let Some(rec) = &recorder {
        let records = rec.take_records();
        println!(
            "diag_records={} drained_total={} dropped={}",
            records.len(),
            drained_total,
            rec.dropped()
        );
        let table = render_summary(&records);
        print!("{table}");
        std::io::stdout().flush().ok();
        if let Some(out_path) = std::env::var_os(DIAG_OUT_ENV) {
            let out_path = out_path.to_string_lossy().into_owned();
            std::fs::write(&out_path, &table)
                .map_err(|e| format!("writing diag summary to {out_path}: {e}"))?;
            let mut raw =
                String::from("index,kernel,rows,cols,quant,weight_bytes,shape,duration_ns\n");
            for r in &records {
                raw.push_str(&format!(
                    "{},{},{},{},{},{},{},{}\n",
                    r.index,
                    r.kernel,
                    r.rows,
                    r.cols,
                    r.quant,
                    r.weight_bytes,
                    r.shape.replace(',', ";"),
                    r.duration_ns
                ));
            }
            let raw_path = format!("{out_path}.records.csv");
            std::fs::write(&raw_path, raw)
                .map_err(|e| format!("writing diag records to {raw_path}: {e}"))?;
            println!("diag_out={out_path} records_out={raw_path}");
        } else {
            println!("diag_out=unset (summary printed only)");
        }
    }
    Ok(())
}

#[cfg(test)]
mod slice_tests {
    use super::resolve_slice_rows;

    /// The slice-size resolution rule, pinned: the env override
    /// applies only inside 1..=1024; anything else falls back to
    /// the 64-row chunk cap the bench protocol runs at.
    #[test]
    fn slice_resolution() {
        assert_eq!(resolve_slice_rows(None), 64);
        assert_eq!(resolve_slice_rows(Some("64")), 64);
        assert_eq!(resolve_slice_rows(Some("128")), 128);
        assert_eq!(resolve_slice_rows(Some("0")), 64);
        assert_eq!(resolve_slice_rows(Some("4096")), 64);
        assert_eq!(resolve_slice_rows(Some("bogus")), 64);
    }
}
