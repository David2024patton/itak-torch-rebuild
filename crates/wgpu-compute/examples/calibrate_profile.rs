//! Calibration: measure this machine's own MMV band
//! table on the local adapter and write it as a
//! calibration profile (see [`wgpu_compute::profile`]),
//! replacing the shipped table's one-card provenance
//! with this card's own sweep.
//!
//! Method: the same two arms the shipped table was
//! derived from (bench/mmv-band-routing-2026-11-06.md,
//! bench/mmv-sweep.sh), run back to back in one
//! process through the session's own routing ladder
//! ([`ProjBench`], so the variant measured is the
//! variant a forward would dispatch):
//!
//! - Control arm: the MMV ceiling pinned closed
//!   (`WGPU_COMPUTE_MMV_MAX_ROWS=1`), K18 and the
//!   chunked form pinned off: every cell measures the
//!   GEMV ladder.
//! - MMV arm: the ceiling open at the kernels' full
//!   band and every band cell pinned to the full band
//!   (`2-16`), so every cell the kernels admit
//!   measures the register-streaming form itself,
//!   not the shipped table's opinion of it.
//!
//! A row is a win for the MMV form under the shipped
//! derivation's rule ([`wgpu_compute::profile::mmv_wins`]):
//! its MMV-arm median is under 0.98x its control
//! median, and both arms' outputs agree with the
//! direct GEMV form (a disagreeing cell is no
//! measurement). Wins adjacent in the sweep group
//! into the cell's ranges
//! ([`wgpu_compute::profile::derive_band_ranges`]).
//! The arms pin every measured knob in the
//! environment, so a profile already on disk cannot
//! contaminate the measurement: env wins over
//! profile per cell by the resolution precedence.
//!
//! The profile is keyed to the adapter and driver
//! that produced it and written to `--out PATH`, or
//! to the standard profile location
//! ([`wgpu_compute::profile::profile_path`]) when no
//! `--out` is given. On a machine with no suitable
//! adapter the run refuses by name and writes
//! nothing: there is no hardware to calibrate. A run
//! below the verdict repetition floor also refuses
//! to write: a profile outlives its run and steers
//! future routing, so it is not derived from a smoke.
//!
//! Usage:
//!   calibrate_profile [--out PATH] [--reps N] [--warmup N]

use std::process::ExitCode;
use std::time::Instant;

use muse_glimmer::Dtype;
use wgpu_compute::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use wgpu_compute::kernels::k1_q4k_gemv::{K1Pipeline, K1Weights};
use wgpu_compute::kernels::k2_q6k_gemv::{K2Pipeline, K2Weights};
use wgpu_compute::kernels::k3_q5k_gemv::{K3Pipeline, K3Weights};
use wgpu_compute::profile::{
    derive_band_ranges, format_band_cell, mmv_wins, write_profile, AdapterKey, CalibrationProfile,
};
use wgpu_compute::timestamps::DispatchMeta;
use wgpu_compute::{
    DetectPolicy, GpuContext, ProjBench, ProjBenchCall, WgpuCompute, WgpuError, MMV_MAX_BATCH,
};

/// Input width of every swept projection: the model's
/// hidden size (the sweep harness's matrix).
const IN: usize = 6656;

/// The swept shapes as (label, output width): the
/// same three the shipped table was derived from.
const SHAPES: [(&str, usize); 3] = [("attn", 6656), ("ffn", 19968), ("kv", 256)];

/// The swept families as (profile name, dtype).
const FAMILIES: [(&str, Dtype); 3] = [
    ("q4k", Dtype::Q4K),
    ("q5k", Dtype::Q5K),
    ("q6k", Dtype::Q6K),
];

/// The row sweep, identical to the sweep harness's:
/// the whole verify band, the batch-9 cliff point,
/// and the matrix threshold at 16.
const ROWS: [u32; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16];

const DEFAULT_REPS: u32 = 25;
const MIN_VERDICT_REPS: u32 = 20;
const DEFAULT_WARMUP: u32 = 3;

/// The nine band env names, composed from the family
/// and shape names exactly as the backend composes
/// its lookups.
fn band_env_names() -> Vec<String> {
    let mut names = Vec::new();
    for (family, _) in FAMILIES {
        for (shape, _) in SHAPES {
            names.push(format!(
                "WGPU_COMPUTE_MMV_{}_{}_ROWS",
                family.to_uppercase(),
                shape.to_uppercase()
            ));
        }
    }
    names
}

/// Pin the process environment for one arm. The two
/// arms differ only in the MMV gate and band cells;
/// every other routing knob the ladder reads is
/// pinned to its shipped-off position in both, so the
/// only variable between arms is the MMV form itself.
fn set_arm_env(mmv_arm: bool) {
    // The ladder's other rungs stay out of both arms.
    std::env::set_var("WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS", "1");
    std::env::set_var("WGPU_COMPUTE_CHUNKED_GEMV_MAX_ROWS", "1");
    std::env::remove_var("WGPU_COMPUTE_WIDE_MIN_ROWS");
    if mmv_arm {
        std::env::set_var("WGPU_COMPUTE_MMV_MAX_ROWS", "16");
        for name in band_env_names() {
            std::env::set_var(name, "2-16");
        }
    } else {
        std::env::set_var("WGPU_COMPUTE_MMV_MAX_ROWS", "1");
        for name in band_env_names() {
            std::env::remove_var(name);
        }
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn lcg_input(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as u32) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn lcg_bytes(seed: u8, count: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(31).wrapping_add(17);
            state
        })
        .collect()
}

/// A synthetic Q4_K tensor's raw bytes (144-byte
/// blocks), the sweep harness's fixture shape:
/// rotating (d, dmin) f16 headers over LCG payloads.
fn synthetic_q4k(in_elems: usize, out_rows: usize) -> Vec<u8> {
    let blocks_per_row = in_elems / 256;
    let mut data = Vec::with_capacity(out_rows * blocks_per_row * 144);
    let headers: [(u16, u16); 4] = [
        (0x3c00, 0x3400),
        (0x3800, 0x0000),
        (0x4000, 0x2e66),
        (0x3e00, 0x3800),
    ];
    for row in 0..out_rows {
        for blk in 0..blocks_per_row {
            let (d, dmin) = headers[(row + blk) % headers.len()];
            let mut block = [0u8; 144];
            block[0..2].copy_from_slice(&d.to_le_bytes());
            block[2..4].copy_from_slice(&dmin.to_le_bytes());
            block[4..144].copy_from_slice(&lcg_bytes((row * 7 + blk * 13 + 5) as u8, 140)[..]);
            data.extend_from_slice(&block);
        }
    }
    data
}

/// A synthetic Q5_K tensor's raw bytes (176-byte
/// blocks, same fixture discipline as Q4_K).
fn synthetic_q5k(in_elems: usize, out_rows: usize) -> Vec<u8> {
    let blocks_per_row = in_elems / 256;
    let mut data = Vec::with_capacity(out_rows * blocks_per_row * 176);
    let headers: [(u16, u16); 4] = [
        (0x3c00, 0x3400),
        (0x3800, 0x0000),
        (0x4000, 0x2e66),
        (0x3e00, 0x3800),
    ];
    for row in 0..out_rows {
        for blk in 0..blocks_per_row {
            let (d, dmin) = headers[(row + blk) % headers.len()];
            let mut block = [0u8; 176];
            block[0..2].copy_from_slice(&d.to_le_bytes());
            block[2..4].copy_from_slice(&dmin.to_le_bytes());
            block[4..176].copy_from_slice(&lcg_bytes((row * 7 + blk * 13 + 5) as u8, 172)[..]);
            data.extend_from_slice(&block);
        }
    }
    data
}

/// A synthetic Q6_K tensor's raw bytes (210-byte
/// blocks, the f16 super-scale at offset 208).
fn synthetic_q6k(in_elems: usize, out_rows: usize) -> Vec<u8> {
    let blocks_per_row = in_elems / 256;
    let mut data = Vec::with_capacity(out_rows * blocks_per_row * 210);
    let ds: [u16; 3] = [0x3c00, 0x3800, 0x4000];
    for row in 0..out_rows {
        for blk in 0..blocks_per_row {
            let mut block = lcg_bytes((row * 11 + blk * 7 + 3) as u8, 210);
            block[208..210].copy_from_slice(&ds[(row + blk) % ds.len()].to_le_bytes());
            data.extend_from_slice(&block);
        }
    }
    data
}

fn quant_name(quant: Dtype) -> &'static str {
    match quant {
        Dtype::Q4K => "Q4_K",
        Dtype::Q5K => "Q5_K",
        Dtype::Q6K => "Q6_K",
        _ => "other",
    }
}

/// The agreement rule between a routed output and
/// the GEMV control output, the sweep harness's own
/// (mirroring the crate's kernel tests): a form that
/// quantizes its activation answers to the 5%
/// quantization band, an f32 form to the tight
/// reassociation bound.
fn agree(got: &[f32], want: &[f32], quantized: bool) -> bool {
    let scale = want.iter().fold(0.0f32, |a, w| a.max(w.abs()));
    let max_diff = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
    if quantized {
        max_diff / scale.max(1.0) <= 0.05
    } else {
        max_diff <= 2e-3 + 5e-5 * scale
    }
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The GEMV control weights, one family at a time,
/// uploaded from the same synthetic bytes the routed
/// handle holds.
enum GemvWeights {
    Q4K(K1Weights),
    Q6K(K2Weights),
    Q5K(K3Weights),
}

/// Time one cell on the wall clock: warmup passes
/// untimed, then one submit-plus-poll per repetition.
/// Returns the median ms. (The sweep harness's
/// timing loop, kept identical so the two tools'
/// medians are the same measurement.)
fn time_wall(ctx: &GpuContext, call: &ProjBenchCall, warmup: u32, reps: u32) -> f64 {
    let run_one = |ctx: &GpuContext| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("calibrate pass"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("calibrate"),
                timestamp_writes: None,
            });
            call.encode_into(&mut pass);
        }
        ctx.queue.submit([encoder.finish()]);
    };
    for _ in 0..warmup {
        run_one(ctx);
    }
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll after warmup");
    let mut times: Vec<f64> = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        let start = Instant::now();
        run_one(ctx);
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll in timed loop");
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    median(&times)
}

/// Time one cell with the diag timestamps system, or
/// `None` when the records do not line up with the
/// repetition count (the caller re-times on the wall
/// clock and says so). Returns the median ms.
fn time_diag(
    ctx: &GpuContext,
    call: &ProjBenchCall,
    meta: DispatchMeta,
    warmup: u32,
    reps: u32,
) -> Option<f64> {
    let recorder = ctx.diag_recorder()?;
    let dropped_before = recorder.dropped();
    let run_one = |ctx: &GpuContext, stamp: bool| {
        let stamp = if stamp {
            ctx.diag_stamp(meta.clone())
        } else {
            None
        };
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("calibrate pass"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("calibrate"),
                timestamp_writes: None,
            });
            let slot = stamp.as_ref().and_then(|s| s.begin(&mut pass));
            call.encode_into(&mut pass);
            if let (Some(s), Some(slot)) = (stamp.as_ref(), slot) {
                s.end(&mut pass, slot);
            }
        }
        ctx.queue.submit([encoder.finish()]);
    };
    for _ in 0..warmup {
        run_one(ctx, false);
    }
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll after warmup");
    for _ in 0..reps {
        run_one(ctx, true);
    }
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll after timed loop");
    recorder.drain(&ctx.device, &ctx.queue).expect("diag drain");
    if recorder.dropped() != dropped_before {
        return None;
    }
    let records = recorder.take_records();
    if records.len() != reps as usize {
        return None;
    }
    let mut times: Vec<f64> = records.iter().map(|r| r.duration_ns as f64 / 1e6).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(median(&times))
}

/// One arm's measurements: per (family, shape, row)
/// the routed variant's median ms and whether its
/// output agreed with the direct GEMV form.
struct Arm {
    medians: [[Vec<f64>; 3]; 3],
    agrees: [[Vec<bool>; 3]; 3],
}

/// Run one arm: every cell of the matrix through the
/// routing ladder of the backend this process just
/// constructed under the arm's env.
fn measure_arm(backend: &WgpuCompute, warmup: u32, reps: u32) -> Arm {
    let ctx = backend.context();
    let timing_diag = ctx.diag_recorder().is_some();
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let max_rows = *ROWS.iter().max().unwrap() as usize;
    let input = lcg_input(7, max_rows * IN);
    let in_buf = ctx.upload_f32("calibrate input", &input);
    let qs_buf = ctx.scratch_buffer("calibrate qs", (qs_words(max_rows, IN) * 4) as u64);
    let ds_buf = ctx.scratch_buffer("calibrate ds", (ds_vec4s(max_rows, IN) * 16) as u64);

    let blank = |fill: f64| -> [[Vec<f64>; 3]; 3] {
        [
            [
                vec![fill; ROWS.len()],
                vec![fill; ROWS.len()],
                vec![fill; ROWS.len()],
            ],
            [
                vec![fill; ROWS.len()],
                vec![fill; ROWS.len()],
                vec![fill; ROWS.len()],
            ],
            [
                vec![fill; ROWS.len()],
                vec![fill; ROWS.len()],
                vec![fill; ROWS.len()],
            ],
        ]
    };
    let mut medians = blank(0.0);
    let mut agrees = [
        [
            vec![false; ROWS.len()],
            vec![false; ROWS.len()],
            vec![false; ROWS.len()],
        ],
        [
            vec![false; ROWS.len()],
            vec![false; ROWS.len()],
            vec![false; ROWS.len()],
        ],
        [
            vec![false; ROWS.len()],
            vec![false; ROWS.len()],
            vec![false; ROWS.len()],
        ],
    ];

    for (fi, (_, quant)) in FAMILIES.iter().enumerate() {
        for (si, (shape_label, out)) in SHAPES.iter().enumerate() {
            let name = format!("calibrate {} {shape_label}", quant_name(*quant));
            let raw = match quant {
                Dtype::Q4K => synthetic_q4k(IN, *out),
                Dtype::Q5K => synthetic_q5k(IN, *out),
                Dtype::Q6K => synthetic_q6k(IN, *out),
                _ => unreachable!("families are the three K quants"),
            };
            let bench = ProjBench::upload(backend, &name, *quant, &raw, IN, *out)
                .expect("proj bench upload");
            let gemv_w = match quant {
                Dtype::Q4K => {
                    GemvWeights::Q4K(k1.upload(ctx, &name, &raw, IN, *out).expect("k1 upload"))
                }
                Dtype::Q6K => {
                    GemvWeights::Q6K(k2.upload(ctx, &name, &raw, IN, *out).expect("k2 upload"))
                }
                Dtype::Q5K => {
                    GemvWeights::Q5K(k3.upload(ctx, &name, &raw, IN, *out).expect("k3 upload"))
                }
                _ => unreachable!("families are the three K quants"),
            };
            let weight_bytes = bench.byte_len() as u64;
            let out_variant =
                ctx.scratch_buffer("calibrate out variant", (max_rows * out * 4) as u64);
            let out_control =
                ctx.scratch_buffer("calibrate out control", (max_rows * out * 4) as u64);

            for (ri, rows_n) in ROWS.iter().enumerate() {
                let run_control = |ctx: &GpuContext| {
                    let mut encoder =
                        ctx.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("calibrate control pass"),
                            });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("calibrate control"),
                            timestamp_writes: None,
                        });
                        match &gemv_w {
                            GemvWeights::Q4K(w) => k1
                                .prepare(ctx, w, &in_buf, &out_control, *rows_n)
                                .expect("k1 prepare")
                                .encode_into(&mut pass),
                            GemvWeights::Q6K(w) => k2
                                .prepare(ctx, w, &in_buf, &out_control, *rows_n)
                                .expect("k2 prepare")
                                .encode_into(&mut pass),
                            GemvWeights::Q5K(w) => k3
                                .prepare(ctx, w, &in_buf, &out_control, *rows_n)
                                .expect("k3 prepare")
                                .encode_into(&mut pass),
                        }
                    }
                    ctx.queue.submit([encoder.finish()]);
                };
                run_control(ctx);
                ctx.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("poll after control");
                let control = ctx
                    .read_f32(&out_control, *rows_n as usize * out)
                    .expect("control readback");

                let call = bench
                    .prepare(
                        backend,
                        &in_buf,
                        &qs_buf,
                        &ds_buf,
                        &out_variant,
                        *rows_n as usize,
                    )
                    .expect("proj bench prepare");
                {
                    let mut encoder =
                        ctx.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("calibrate check pass"),
                            });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("calibrate check"),
                            timestamp_writes: None,
                        });
                        call.encode_into(&mut pass);
                    }
                    ctx.queue.submit([encoder.finish()]);
                }
                ctx.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("poll after check");
                let got = ctx
                    .read_f32(&out_variant, *rows_n as usize * out)
                    .expect("variant readback");
                agrees[fi][si][ri] = agree(&got, &control, call.quantized());

                let meta = DispatchMeta {
                    kernel: call.kernel_name(),
                    rows: u64::from(*rows_n),
                    cols: IN as u64,
                    quant: quant_name(*quant),
                    weight_bytes,
                    shape: format!("calibrate {shape_label}"),
                };
                let median_ms = if timing_diag {
                    match time_diag(ctx, &call, meta, warmup, reps) {
                        Some(t) => t,
                        None => time_wall(ctx, &call, warmup, reps),
                    }
                } else {
                    time_wall(ctx, &call, warmup, reps)
                };
                medians[fi][si][ri] = median_ms;
            }
        }
    }
    Arm { medians, agrees }
}

/// Open the backend for one arm, or refuse by name:
/// calibration needs the hardware it calibrates for,
/// so no suitable adapter is a refusal (exit 1), not
/// a skip, and nothing is written.
fn open_backend(arm: &str) -> Result<WgpuCompute, ExitCode> {
    match WgpuCompute::with_policy(&DetectPolicy::default()) {
        Ok(backend) => Ok(backend),
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "REFUSED: no suitable GPU on this machine to calibrate \
                 (arm {arm}; enumerated: [{}]); nothing written",
                seen.join("; ")
            );
            Err(ExitCode::from(1))
        }
        Err(e) => {
            eprintln!("backend construction failed (arm {arm}): {e}");
            Err(ExitCode::from(2))
        }
    }
}

fn run() -> Result<(), ExitCode> {
    let args: Vec<String> = std::env::args().collect();
    let reps: u32 = arg(&args, "--reps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPS);
    let warmup: u32 = arg(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARMUP);
    if reps < MIN_VERDICT_REPS {
        eprintln!(
            "REFUSED: reps {reps} is below the {MIN_VERDICT_REPS}-repetition \
             verdict floor; a profile is not derived from a smoke. \
             Pass --reps {DEFAULT_REPS} (the default) for a verdict run; \
             nothing written"
        );
        return Err(ExitCode::from(1));
    }
    // Diag timestamps are the timing method when the
    // device grants them (the sweep harness's method);
    // set the mode unless the caller already chose one.
    if std::env::var_os("WGPU_COMPUTE_DIAG").is_none() {
        std::env::set_var("WGPU_COMPUTE_DIAG", "timing");
    }

    set_arm_env(false);
    let control_backend = open_backend("control")?;
    let key = AdapterKey::from_facts(&control_backend.context().adapter);
    let adapter_name = control_backend.context().adapter.name.clone();
    println!("adapter: {adapter_name}");
    println!("arm control: GEMV ladder (MMV ceiling pinned closed)");
    let control = measure_arm(&control_backend, warmup, reps);
    drop(control_backend);

    set_arm_env(true);
    let mmv_backend = open_backend("mmv")?;
    println!("arm mmv: register-streaming forms (ceiling and bands pinned open)");
    let mmv = measure_arm(&mmv_backend, warmup, reps);
    drop(mmv_backend);

    let mut profile = CalibrationProfile::new(key);
    profile.mmv_max_rows = Some(MMV_MAX_BATCH);
    println!("derived band table (win = MMV median < 0.98x control, both arms agreeing):");
    for (fi, (family, _)) in FAMILIES.iter().enumerate() {
        for (si, (shape, _)) in SHAPES.iter().enumerate() {
            let wins: Vec<bool> = ROWS
                .iter()
                .enumerate()
                .map(|(ri, _)| {
                    control.agrees[fi][si][ri]
                        && mmv.agrees[fi][si][ri]
                        && mmv_wins(mmv.medians[fi][si][ri], control.medians[fi][si][ri])
                })
                .collect();
            let cell = format_band_cell(&derive_band_ranges(&ROWS, &wins));
            println!("  {family} {shape}: {cell}");
            let placed = profile.set_band_cell_named(family, shape, cell);
            assert!(placed, "family and shape names are the table's own");
        }
    }

    let out_path = match arg(&args, "--out") {
        Some(p) => std::path::PathBuf::from(p),
        None => match wgpu_compute::profile::profile_path() {
            Some(p) => p,
            None => {
                eprintln!(
                    "REFUSED: no profile location resolves (no --out, no \
                     WGPU_COMPUTE_PROFILE_PATH, no XDG_CONFIG_HOME or HOME); \
                     nothing written"
                );
                return Err(ExitCode::from(1));
            }
        },
    };
    if let Err(e) = write_profile(&out_path, &profile) {
        eprintln!(
            "REFUSED: writing the profile to {} failed: {e}",
            out_path.display()
        );
        return Err(ExitCode::from(1));
    }
    println!(
        "wrote {} (keyed to [{}])",
        out_path.display(),
        profile.key.describe()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}
