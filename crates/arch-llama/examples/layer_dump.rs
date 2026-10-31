//! Per-layer activation dump (layer-diff harness, follow-up to
//! integration run A).
//!
//! Runs one prefill of fixed token ids through
//! [`LlamaModel::forward_traced`] and writes every captured
//! intermediate as a raw little-endian f32 `.bin` plus a
//! `manifest.json` of shapes, in the same naming scheme the
//! llama.cpp-side dumper (`golden/tools/cb_dump.cpp`) produces, so
//! `golden/tools/compare_layers.py` can diff the two directories
//! tensor by tensor.
//!
//! Usage:
//!   layer_dump --model <file.gguf> --out <dir> --ids 16,22,488,220

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arch_llama::{ForwardTrace, LlamaConfig, LlamaModel, Weights};
use gguf::GgufFile;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("layer_dump: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn next_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next()
        .ok_or_else(|| format!("flag {flag} needs a value"))
}

struct Dumper {
    dir: PathBuf,
    manifest: String,
}

impl Dumper {
    fn new(dir: &Path) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Dumper {
            dir: dir.to_path_buf(),
            manifest: String::from("{\n"),
        })
    }

    fn tensor(&mut self, name: &str, data: &[f32], width: usize) -> Result<(), String> {
        let rows = data.len().checked_div(width).unwrap_or(0);
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let path = self.dir.join(format!("{name}.bin"));
        fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        self.manifest
            .push_str(&format!("  \"{name}\": [{width}, {rows}],\n"));
        Ok(())
    }

    fn finish(mut self, meta: &str) -> Result<(), String> {
        if self.manifest.ends_with(",\n") {
            self.manifest.truncate(self.manifest.len() - 2);
            self.manifest.push('\n');
        }
        self.manifest.push_str("}\n");
        let path = self.dir.join("manifest.json");
        fs::write(&path, self.manifest).map_err(|e| format!("{}: {e}", path.display()))?;
        let path = self.dir.join("meta.txt");
        fs::write(&path, meta).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(())
    }
}

fn dump_trace(
    trace: &ForwardTrace,
    logits: &[f32],
    cfg: &LlamaConfig,
    dir: &Path,
    ids: &[u32],
) -> Result<(), String> {
    let rows = ids.len();
    let hidden = cfg.hidden as usize;
    let mut d = Dumper::new(dir)?;
    d.tensor("embd", &trace.embd, hidden)?;
    for (il, lt) in trace.layers.iter().enumerate() {
        let p = format!("L{il:02}");
        d.tensor(&format!("{p}_attn_norm"), &lt.attn_norm, hidden)?;
        d.tensor(
            &format!("{p}_q_pre_rope"),
            &lt.q_pre_rope,
            cfg.q_width() as usize,
        )?;
        d.tensor(
            &format!("{p}_k_pre_rope"),
            &lt.k_pre_rope,
            cfg.k_width() as usize,
        )?;
        d.tensor(&format!("{p}_q"), &lt.q, cfg.q_width() as usize)?;
        d.tensor(&format!("{p}_k"), &lt.k, cfg.k_width() as usize)?;
        d.tensor(&format!("{p}_v"), &lt.v, cfg.v_width() as usize)?;
        d.tensor(&format!("{p}_ffn_inp"), &lt.ffn_inp, hidden)?;
        d.tensor(&format!("{p}_ffn_norm"), &lt.ffn_norm, hidden)?;
        d.tensor(&format!("{p}_ffn_out"), &lt.ffn_out, hidden)?;
        d.tensor(&format!("{p}_l_out"), &lt.l_out, hidden)?;
    }
    d.tensor("result_norm", &trace.result_norm, hidden)?;
    // Logits are one row of vocab width for the final position.
    d.tensor("logits", logits, cfg.vocab_size as usize)?;
    let _ = rows;
    d.finish(&format!(
        "engine=arch-llama ids={ids:?} rows={rows} hidden={hidden} layers={} q_width={} k_width={} rope_base={} rope_dim={} eps={}\n",
        cfg.block_count,
        cfg.q_width(),
        cfg.k_width(),
        cfg.rope_freq_base,
        cfg.rope_dim,
        cfg.rms_eps,
    ))
}

fn run() -> Result<(), String> {
    let mut model_path = None;
    let mut out_dir = None;
    let mut ids: Vec<u32> = Vec::new();
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model_path = Some(next_value(&mut it, &arg)?),
            "--out" => out_dir = Some(next_value(&mut it, &arg)?),
            "--ids" => {
                let v = next_value(&mut it, &arg)?;
                ids = v
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().map_err(|_| format!("bad id {s}")))
                    .collect::<Result<_, _>>()?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let model_path = model_path.ok_or("missing --model")?;
    let out_dir = out_dir.ok_or("missing --out")?;
    if ids.is_empty() {
        return Err("missing --ids".to_string());
    }

    let file =
        GgufFile::open(Path::new(&model_path)).map_err(|e| format!("open {model_path}: {e}"))?;
    let cfg = LlamaConfig::from_file(&file).map_err(|e| format!("config: {e}"))?;
    let weights = Weights::load(&file, &cfg).map_err(|e| format!("weights: {e}"))?;
    let model = LlamaModel::new(cfg.clone(), weights);
    let mut cache = model.new_cache().map_err(|e| e.to_string())?;
    let (logits, trace) = model
        .forward_traced(&ids, &mut cache)
        .map_err(|e| format!("forward: {e}"))?;
    dump_trace(&trace, &logits, &cfg, Path::new(&out_dir), &ids)?;
    let mut top = 0usize;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > logits[top] {
            top = i;
        }
    }
    eprintln!(
        "layer_dump: {} layers, ids {ids:?}, top-1 logit id {top} ({:.4})",
        cfg.block_count, logits[top]
    );
    Ok(())
}
