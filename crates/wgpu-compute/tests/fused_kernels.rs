//! The fusion kernels against the unfused compositions they
//! replace in the device-resident session: same inputs, same
//! math, fewer dispatches. K9 (the fused attention
//! preparation) is checked against the session's original
//! chain step for step: K4's Q and K norms, K5's rotations on
//! sliding layers, K5's append into the KV store, in the
//! chain's own order, from the same pipelines.
//!
//! These tests need a GPU. The numeric tests take the
//! backend's default policy, a discrete Vulkan GPU, and skip
//! with a printed reason where none qualifies, exactly like
//! the other kernel suites. On David's server the 7900 XTX
//! answers and every test runs for real.

use muse_glimmer::Dtype;
use wgpu_compute::kernels::k1_q4k_gemv::{
    f32_to_f16_bits, shipped_shape as k1_shipped, K1ActGeom, K1Pipeline, ACT_GATED, ACT_SWIGLU,
};
use wgpu_compute::kernels::k2_q6k_gemv::{shipped_shape as k2_shipped, K2Pipeline};
use wgpu_compute::kernels::k3_q5k_gemv::{K3Pipeline, FINISH_SCALE, FINISH_SOFTCAP};
use wgpu_compute::kernels::k4_rmsnorm::K4Pipeline;
use wgpu_compute::kernels::k5_rope_kv::K5Pipeline;
use wgpu_compute::kernels::k6_attention::{AttendGeom, K6Pipeline, LayerKv};
use wgpu_compute::kernels::k8_head::K8Pipeline;
use wgpu_compute::kernels::k9_attn_prep::{K9Pipeline, PrepArgs};
use wgpu_compute::{WgpuCompute, WgpuError};

/// A backend, or `None` after printing why this machine cannot
/// provide one (the K1 tests' pattern).
fn backend_or_skip() -> Option<WgpuCompute> {
    match WgpuCompute::new() {
        Ok(b) => Some(b),
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping GPU test: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            None
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    }
}

/// Deterministic pseudo-random values in [-0.5, 0.5).
fn lcg(seed: u64, count: usize) -> Vec<f32> {
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

/// The largest absolute difference between two equal-length
/// vectors (panics on a length mismatch).
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in comparison");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// One case's geometry and inputs, shared by the reference
/// composition and the fused runs.
struct PrepCase {
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    start: usize,
    ring: Option<usize>,
    base: f32,
    eps: f32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_w: Vec<f32>,
    k_w: Vec<f32>,
}

impl PrepCase {
    fn new(sliding: bool, seed: u64) -> Self {
        let rows = 3;
        let q_heads = 4;
        let kv_heads = 2;
        let head_dim = 64;
        PrepCase {
            rows,
            q_heads,
            kv_heads,
            head_dim,
            // A nonzero start with a small ring wraps the
            // append slots (7, 8, 9 -> 7, 0, 1) in the sliding
            // case.
            start: 7,
            ring: sliding.then_some(8),
            base: 500_000.0,
            eps: 1e-5,
            q: lcg(seed, rows * q_heads * head_dim),
            k: lcg(seed + 1, rows * kv_heads * head_dim),
            v: lcg(seed + 2, rows * kv_heads * head_dim),
            q_w: lcg(seed + 3, head_dim).iter().map(|x| x + 1.0).collect(),
            k_w: lcg(seed + 4, head_dim).iter().map(|x| x + 1.0).collect(),
        }
    }

    fn q_width(&self) -> usize {
        self.q_heads * self.head_dim
    }

    fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// KV store capacity the case needs: the ring for sliding
    /// layers, the end position for global ones.
    fn capacity(&self) -> usize {
        self.ring.unwrap_or(self.start + self.rows)
    }
}

/// The reference chain: K4 norms, K5 rotations (sliding),
/// K5 append, in the resident session's original order.
/// Returns (prepared Q rows, KV keys, KV values).
fn reference_prep(gpu: &WgpuCompute, case: &PrepCase) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ctx = gpu.context();
    let k4 = K4Pipeline::new(&ctx.device).expect("k4 pipeline");
    let k5 = K5Pipeline::new(&ctx.device).expect("k5 pipeline");
    let q_in = ctx.upload_f32("ref q", &case.q);
    let k_in = ctx.upload_f32("ref k", &case.k);
    let v_in = ctx.upload_f32("ref v", &case.v);
    let q_w = ctx.upload_f32("ref q w", &case.q_w);
    let k_w = ctx.upload_f32("ref k w", &case.k_w);
    let q_normed = ctx.scratch_buffer("ref q normed", (case.q.len() * 4) as u64);
    let k_normed = ctx.scratch_buffer("ref k normed", (case.k.len() * 4) as u64);
    let cap_elems = case.capacity() * case.kv_width();
    let kv_k = ctx.scratch_buffer("ref kv k", (cap_elems * 4) as u64);
    let kv_v = ctx.scratch_buffer("ref kv v", (cap_elems * 4) as u64);

    let qn = k4.prepare_rms_norm(
        ctx,
        &q_in,
        &q_w,
        &q_normed,
        case.rows * case.q_heads,
        case.head_dim,
        case.eps,
    );
    let kn = k4.prepare_rms_norm(
        ctx,
        &k_in,
        &k_w,
        &k_normed,
        case.rows * case.kv_heads,
        case.head_dim,
        case.eps,
    );
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    qn.encode(&mut encoder);
    kn.encode(&mut encoder);
    if case.ring.is_some() {
        let rq = k5
            .prepare_rotate(
                ctx,
                &q_normed,
                case.rows,
                case.q_heads,
                case.head_dim,
                case.start,
                case.base,
            )
            .expect("rotate q");
        let rk = k5
            .prepare_rotate(
                ctx,
                &k_normed,
                case.rows,
                case.kv_heads,
                case.head_dim,
                case.start,
                case.base,
            )
            .expect("rotate k");
        rq.encode(&mut encoder);
        rk.encode(&mut encoder);
    }
    let append = k5
        .prepare_append(
            ctx,
            &k_normed,
            &v_in,
            &kv_k,
            &kv_v,
            case.rows,
            case.kv_width(),
            case.start,
            case.ring,
        )
        .expect("append");
    append.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    (
        ctx.read_f32(&q_normed, case.q.len()).expect("read q"),
        ctx.read_f32(&kv_k, cap_elems).expect("read kv k"),
        ctx.read_f32(&kv_v, cap_elems).expect("read kv v"),
    )
}

/// The fused kernel over separate projection buffers (the
/// session's Stage A form). Same outputs as the reference.
fn fused_prep_separate(gpu: &WgpuCompute, case: &PrepCase) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ctx = gpu.context();
    let k9 = K9Pipeline::new(&ctx.device).expect("k9 pipeline");
    let q_in = ctx.upload_f32("fused q", &case.q);
    let k_in = ctx.upload_f32("fused k", &case.k);
    let v_in = ctx.upload_f32("fused v", &case.v);
    let q_w = ctx.upload_f32("fused q w", &case.q_w);
    let k_w = ctx.upload_f32("fused k w", &case.k_w);
    let q_out = ctx.scratch_buffer("fused q out", (case.q.len() * 4) as u64);
    let cap_elems = case.capacity() * case.kv_width();
    let kv_k = ctx.scratch_buffer("fused kv k", (cap_elems * 4) as u64);
    let kv_v = ctx.scratch_buffer("fused kv v", (cap_elems * 4) as u64);
    let prep = k9
        .prepare(
            ctx,
            &PrepArgs {
                q_in: &q_in,
                q_stride: case.q_width(),
                q_off: 0,
                k_in: &k_in,
                kv_stride: case.kv_width(),
                k_off: 0,
                v_in: &v_in,
                v_stride: case.kv_width(),
                v_off: 0,
                q_w: &q_w,
                k_w: &k_w,
                q_out: &q_out,
                kv_k: &kv_k,
                kv_v: &kv_v,
                rows: case.rows,
                q_heads: case.q_heads,
                kv_heads: case.kv_heads,
                head_dim: case.head_dim,
                start_pos: case.start,
                ring: case.ring,
                rotate: case.ring.is_some(),
                base: case.base,
                eps: case.eps,
            },
        )
        .expect("prepare prep");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    prep.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    (
        ctx.read_f32(&q_out, case.q.len()).expect("read q"),
        ctx.read_f32(&kv_k, cap_elems).expect("read kv k"),
        ctx.read_f32(&kv_v, cap_elems).expect("read kv v"),
    )
}

/// The fused kernel over one concatenated QKV buffer (the
/// layout the batched QKV projection writes): per row, the Q
/// segment, then K, then V.
fn fused_prep_concat(gpu: &WgpuCompute, case: &PrepCase) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ctx = gpu.context();
    let k9 = K9Pipeline::new(&ctx.device).expect("k9 pipeline");
    let width = case.q_width() + 2 * case.kv_width();
    let mut concat = vec![0.0f32; case.rows * width];
    for r in 0..case.rows {
        let dst = r * width;
        let qs = r * case.q_width();
        let ks = r * case.kv_width();
        concat[dst..dst + case.q_width()].copy_from_slice(&case.q[qs..qs + case.q_width()]);
        concat[dst + case.q_width()..dst + case.q_width() + case.kv_width()]
            .copy_from_slice(&case.k[ks..ks + case.kv_width()]);
        concat[dst + case.q_width() + case.kv_width()..dst + width]
            .copy_from_slice(&case.v[ks..ks + case.kv_width()]);
    }
    let qkv = ctx.upload_f32("fused qkv", &concat);
    let q_w = ctx.upload_f32("fused q w", &case.q_w);
    let k_w = ctx.upload_f32("fused k w", &case.k_w);
    let q_out = ctx.scratch_buffer("fused q out", (case.q.len() * 4) as u64);
    let cap_elems = case.capacity() * case.kv_width();
    let kv_k = ctx.scratch_buffer("fused kv k", (cap_elems * 4) as u64);
    let kv_v = ctx.scratch_buffer("fused kv v", (cap_elems * 4) as u64);
    let prep = k9
        .prepare(
            ctx,
            &PrepArgs {
                q_in: &qkv,
                q_stride: width,
                q_off: 0,
                k_in: &qkv,
                kv_stride: width,
                k_off: case.q_width(),
                v_in: &qkv,
                v_stride: width,
                v_off: case.q_width() + case.kv_width(),
                q_w: &q_w,
                k_w: &k_w,
                q_out: &q_out,
                kv_k: &kv_k,
                kv_v: &kv_v,
                rows: case.rows,
                q_heads: case.q_heads,
                kv_heads: case.kv_heads,
                head_dim: case.head_dim,
                start_pos: case.start,
                ring: case.ring,
                rotate: case.ring.is_some(),
                base: case.base,
                eps: case.eps,
            },
        )
        .expect("prepare prep");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    prep.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    (
        ctx.read_f32(&q_out, case.q.len()).expect("read q"),
        ctx.read_f32(&kv_k, cap_elems).expect("read kv k"),
        ctx.read_f32(&kv_v, cap_elems).expect("read kv v"),
    )
}

/// The fused kernel over the partial-batch layout (the
/// layout a Q/K/gate batch writes): Q and K ride a wide
/// concatenated buffer (with a dummy gate-width tail the
/// kernel never reads), V keeps its own buffer and stride.
fn fused_prep_split(gpu: &WgpuCompute, case: &PrepCase) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ctx = gpu.context();
    let k9 = K9Pipeline::new(&ctx.device).expect("k9 pipeline");
    let width = case.q_width() + case.kv_width() + case.q_width();
    let mut concat = vec![0.0f32; case.rows * width];
    for r in 0..case.rows {
        let dst = r * width;
        let qs = r * case.q_width();
        let ks = r * case.kv_width();
        concat[dst..dst + case.q_width()].copy_from_slice(&case.q[qs..qs + case.q_width()]);
        concat[dst + case.q_width()..dst + case.q_width() + case.kv_width()]
            .copy_from_slice(&case.k[ks..ks + case.kv_width()]);
    }
    let qk = ctx.upload_f32("fused qk", &concat);
    let v_in = ctx.upload_f32("fused v in", &case.v);
    let q_w = ctx.upload_f32("fused q w", &case.q_w);
    let k_w = ctx.upload_f32("fused k w", &case.k_w);
    let q_out = ctx.scratch_buffer("fused q out", (case.q.len() * 4) as u64);
    let cap_elems = case.capacity() * case.kv_width();
    let kv_k = ctx.scratch_buffer("fused kv k", (cap_elems * 4) as u64);
    let kv_v = ctx.scratch_buffer("fused kv v", (cap_elems * 4) as u64);
    let prep = k9
        .prepare(
            ctx,
            &PrepArgs {
                q_in: &qk,
                q_stride: width,
                q_off: 0,
                k_in: &qk,
                kv_stride: width,
                k_off: case.q_width(),
                v_in: &v_in,
                v_stride: case.kv_width(),
                v_off: 0,
                q_w: &q_w,
                k_w: &k_w,
                q_out: &q_out,
                kv_k: &kv_k,
                kv_v: &kv_v,
                rows: case.rows,
                q_heads: case.q_heads,
                kv_heads: case.kv_heads,
                head_dim: case.head_dim,
                start_pos: case.start,
                ring: case.ring,
                rotate: case.ring.is_some(),
                base: case.base,
                eps: case.eps,
            },
        )
        .expect("prepare prep");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    prep.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    (
        ctx.read_f32(&q_out, case.q.len()).expect("read q"),
        ctx.read_f32(&kv_k, cap_elems).expect("read kv k"),
        ctx.read_f32(&kv_v, cap_elems).expect("read kv v"),
    )
}

fn check_prep(case: &PrepCase) {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let (want_q, want_k, want_v) = reference_prep(&gpu, case);
    type PrepRun = fn(&WgpuCompute, &PrepCase) -> (Vec<f32>, Vec<f32>, Vec<f32>);
    for (name, run) in [
        ("separate", fused_prep_separate as PrepRun),
        ("concat", fused_prep_concat as PrepRun),
        ("split", fused_prep_split as PrepRun),
    ] {
        let (got_q, got_k, got_v) = run(&gpu, case);
        let dq = max_abs(&got_q, &want_q);
        let dk = max_abs(&got_k, &want_k);
        let dv = max_abs(&got_v, &want_v);
        eprintln!(
            "k9 prep ({name}, sliding={}): q {dq:.3e}, kv k {dk:.3e}, kv v {dv:.3e}",
            case.ring.is_some()
        );
        assert!(dq < 1e-5, "prepared Q drifted ({name}): {dq}");
        assert!(dk < 1e-5, "appended K drifted ({name}): {dk}");
        assert_eq!(got_v, want_v, "appended V must be an exact copy ({name})");
    }
}

#[test]
fn k9_prep_matches_the_k4_k5_chain_on_sliding_layers() {
    check_prep(&PrepCase::new(true, 91));
}

#[test]
fn k9_prep_matches_the_k4_k5_chain_on_global_layers() {
    check_prep(&PrepCase::new(false, 97));
}

/// Run K6 two-pass and flash over the same resident KV and
/// compare: the flash form must reproduce the two-pass
/// output (its operation sequence per element is the
/// two-pass sequence; only the score storage differs).
fn check_flash(geom: AttendGeom, seed: u64) {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let k6 = K6Pipeline::new(&ctx.device).expect("k6 pipeline");
    let q_width = geom.q_heads * geom.head_dim;
    let kv_width = geom.kv_heads * geom.head_dim;
    let capacity = geom.ring.unwrap_or(geom.first_pos + geom.stored);
    let q = ctx.upload_f32("flash q", &lcg(seed, geom.rows * q_width));
    let kv = LayerKv {
        k: ctx.upload_f32("flash kv k", &lcg(seed + 1, capacity * kv_width)),
        v: ctx.upload_f32("flash kv v", &lcg(seed + 2, capacity * kv_width)),
        capacity,
    };
    let scores_buf = ctx.scratch_buffer(
        "flash scores",
        (geom.rows * geom.q_heads * geom.stored * 4) as u64,
    );
    let out_two = ctx.scratch_buffer("flash out two", (geom.rows * q_width * 4) as u64);
    let out_flash = ctx.scratch_buffer("flash out fused", (geom.rows * q_width * 4) as u64);
    let scores = k6
        .prepare_scores(ctx, &q, &kv, &scores_buf, &geom)
        .expect("prepare scores");
    let reduce = k6
        .prepare_reduce(ctx, &scores_buf, &kv, &out_two, &geom)
        .expect("prepare reduce");
    let flash = k6
        .prepare_flash(ctx, &q, &kv, &out_flash, &geom)
        .expect("prepare flash");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        scores.encode_into(&mut pass);
        reduce.encode_into(&mut pass);
        flash.encode_into(&mut pass);
    }
    ctx.queue.submit([encoder.finish()]);
    let want = ctx
        .read_f32(&out_two, geom.rows * q_width)
        .expect("read two-pass");
    let got = ctx
        .read_f32(&out_flash, geom.rows * q_width)
        .expect("read flash");
    let d = max_abs(&got, &want);
    eprintln!(
        "k6 flash (rows={}, stored={}, window={:?}): max abs diff {d:.3e}",
        geom.rows, geom.stored, geom.window
    );
    assert!(d < 1e-6, "flash attention drifted from two-pass: {d}");
}

#[test]
fn k6_flash_matches_two_pass() {
    // Global layer, decode row, a stored count that crosses
    // the 256-key tile boundary.
    check_flash(
        AttendGeom {
            rows: 1,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            stored: 300,
            first_pos: 0,
            start_pos: 299,
            window: None,
            ring: None,
        },
        31,
    );
    // Windowed layer: the mask must bite (queries see only
    // the last 16 positions) and slots wrap the ring.
    check_flash(
        AttendGeom {
            rows: 1,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            stored: 64,
            first_pos: 40,
            start_pos: 103,
            window: Some(16),
            ring: Some(64),
        },
        33,
    );
    // A short stored count under one tile, multi-row.
    check_flash(
        AttendGeom {
            rows: 3,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            stored: 37,
            first_pos: 5,
            start_pos: 5,
            window: None,
            ring: None,
        },
        35,
    );
}

#[test]
fn k4_norm_chain_matches_the_two_dispatch_composition() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let k4 = K4Pipeline::new(&ctx.device).expect("k4 pipeline");
    // A width that strides the lanes unevenly (6656 = the
    // real hidden width: 26 elements per lane exactly; use
    // 6656 and an awkward 4100 in two cases).
    for (rows, width) in [(2usize, 6656usize), (3, 4100)] {
        let residual = lcg(51, rows * width);
        let branch = lcg(52, rows * width);
        let w_post: Vec<f32> = lcg(53, width).iter().map(|x| x + 1.0).collect();
        let w_pre: Vec<f32> = lcg(54, width).iter().map(|x| x + 1.0).collect();
        let res_buf = ctx.upload_f32("chain residual", &residual);
        let bra_buf = ctx.upload_f32("chain branch", &branch);
        let wp_buf = ctx.upload_f32("chain w post", &w_post);
        let wpre_buf = ctx.upload_f32("chain w pre", &w_pre);
        let stream_ref = ctx.scratch_buffer("chain stream ref", (rows * width * 4) as u64);
        let normed_ref = ctx.scratch_buffer("chain normed ref", (rows * width * 4) as u64);
        let stream_new = ctx.scratch_buffer("chain stream new", (rows * width * 4) as u64);
        let normed_new = ctx.scratch_buffer("chain normed new", (rows * width * 4) as u64);
        let post = k4.prepare_norm_residual(
            ctx,
            &res_buf,
            &bra_buf,
            &wp_buf,
            &stream_ref,
            rows,
            width,
            1e-8,
        );
        let pre = k4.prepare_rms_norm(ctx, &stream_ref, &wpre_buf, &normed_ref, rows, width, 1e-5);
        let chain = k4
            .prepare_norm_chain(
                ctx,
                &res_buf,
                &bra_buf,
                &wp_buf,
                &wpre_buf,
                &stream_new,
                &normed_new,
                rows,
                width,
                1e-8,
                1e-5,
            )
            .expect("prepare chain");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            post.encode_into(&mut pass);
            pre.encode_into(&mut pass);
            chain.encode_into(&mut pass);
        }
        ctx.queue.submit([encoder.finish()]);
        let want_s = ctx
            .read_f32(&stream_ref, rows * width)
            .expect("read stream");
        let got_s = ctx
            .read_f32(&stream_new, rows * width)
            .expect("read stream");
        let want_n = ctx
            .read_f32(&normed_ref, rows * width)
            .expect("read normed");
        let got_n = ctx
            .read_f32(&normed_new, rows * width)
            .expect("read normed");
        // The stream is the same operation sequence as the
        // two-dispatch form; the fused kernel's codegen may
        // contract a multiply-add the separate kernel does
        // not, so equality is ULP-class, not bitwise.
        let ds = max_abs(&got_s, &want_s);
        eprintln!("k4 chain (width {width}): stream max abs diff {ds:.3e}");
        assert!(ds < 2e-6, "chain stream drifted (width {width}): {ds}");
        let d = max_abs(&got_n, &want_n);
        eprintln!("k4 chain (width {width}): normed max abs diff {d:.3e}");
        assert!(d < 1e-6, "chain normed output drifted (width {width}): {d}");
    }
}

#[test]
fn k4_norm_close_matches_the_composition_and_emits_the_head_row() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let k4 = K4Pipeline::new(&ctx.device).expect("k4 pipeline");
    let (rows, width) = (3usize, 6656usize);
    let residual = lcg(81, rows * width);
    let branch = lcg(82, rows * width);
    let w_post: Vec<f32> = lcg(83, width).iter().map(|x| x + 1.0).collect();
    let w_final: Vec<f32> = lcg(84, width).iter().map(|x| x + 1.0).collect();
    let res_buf = ctx.upload_f32("close residual", &residual);
    let bra_buf = ctx.upload_f32("close branch", &branch);
    let wp_buf = ctx.upload_f32("close w post", &w_post);
    let wf_buf = ctx.upload_f32("close w final", &w_final);
    let stream_ref = ctx.scratch_buffer("close stream ref", (rows * width * 4) as u64);
    let last_ref = ctx.scratch_buffer("close last ref", (width * 4) as u64);
    let head_ref = ctx.scratch_buffer("close head ref", (width * 4) as u64);
    let stream_new = ctx.scratch_buffer("close stream new", (rows * width * 4) as u64);
    let head_new = ctx.scratch_buffer("close head new", (width * 4) as u64);
    let post = k4.prepare_norm_residual(
        ctx,
        &res_buf,
        &bra_buf,
        &wp_buf,
        &stream_ref,
        rows,
        width,
        1e-8,
    );
    let fin = k4.prepare_rms_norm(ctx, &last_ref, &wf_buf, &head_ref, 1, width, 1e-5);
    let close = k4
        .prepare_norm_close(
            ctx,
            &res_buf,
            &bra_buf,
            &wp_buf,
            &wf_buf,
            &stream_new,
            &head_new,
            rows,
            width,
            1e-8,
            1e-5,
        )
        .expect("prepare close");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        post.encode_into(&mut pass);
        close.encode_into(&mut pass);
    }
    // The reference's last-row copy, between the passes.
    encoder.copy_buffer_to_buffer(
        &stream_ref,
        ((rows - 1) * width * 4) as u64,
        &last_ref,
        0,
        (width * 4) as u64,
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        fin.encode_into(&mut pass);
    }
    ctx.queue.submit([encoder.finish()]);
    let want_s = ctx
        .read_f32(&stream_ref, rows * width)
        .expect("read stream");
    let got_s = ctx
        .read_f32(&stream_new, rows * width)
        .expect("read stream");
    let want_h = ctx.read_f32(&head_ref, width).expect("read head");
    let got_h = ctx.read_f32(&head_new, width).expect("read head");
    let ds = max_abs(&got_s, &want_s);
    let dh = max_abs(&got_h, &want_h);
    eprintln!("k4 close: stream max abs diff {ds:.3e}, head max abs diff {dh:.3e}");
    assert!(ds < 2e-6, "close stream drifted: {ds}");
    assert!(dh < 1e-6, "close head row drifted: {dh}");
}

/// A Q4_K tensor's raw bytes, by-construction blocks like
/// the cross-backend fixture's: small scales, deterministic
/// nibbles.
fn q4k_bytes(seed: u64, inn: usize, out: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 56) as u8
    };
    let blocks = inn * out / 256;
    let mut data = Vec::with_capacity(blocks * 144);
    for _ in 0..blocks {
        data.extend_from_slice(&f32_to_f16_bits(0.005).to_le_bytes());
        data.extend_from_slice(&f32_to_f16_bits(0.002).to_le_bytes());
        for _ in 0..12 {
            data.push(next() & 0x07);
        }
        for _ in 0..128 {
            data.push(next());
        }
    }
    data
}

/// A Q6_K tensor's raw bytes: any byte pattern dequantizes,
/// so deterministic pseudo-random planes with a small scale
/// serve (the values stay in a sane range).
fn q6k_bytes(seed: u64, inn: usize, out: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 56) as u8
    };
    let blocks = inn * out / 256;
    let mut data = Vec::with_capacity(blocks * 210);
    for _ in 0..blocks {
        for _ in 0..128 {
            data.push(next());
        }
        for _ in 0..64 {
            data.push(next());
        }
        for _ in 0..16 {
            data.push(next() & 0x03);
        }
        data.extend_from_slice(&f32_to_f16_bits(0.01).to_le_bytes());
    }
    data
}

/// A Q5_K tensor's raw bytes: 176-byte blocks (d, dmin, 12
/// scale bytes, 32 high-bit bytes, 128 low-bit bytes),
/// deterministic payload like the other builders.
fn q5k_bytes(seed: u64, inn: usize, out: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 56) as u8
    };
    let blocks = inn * out / 256;
    let mut data = Vec::with_capacity(blocks * 176);
    for _ in 0..blocks {
        data.extend_from_slice(&f32_to_f16_bits(0.005).to_le_bytes());
        data.extend_from_slice(&f32_to_f16_bits(0.002).to_le_bytes());
        for _ in 0..12 {
            data.push(next() & 0x07);
        }
        for _ in 0..32 {
            data.push(next());
        }
        for _ in 0..128 {
            data.push(next());
        }
    }
    data
}

#[test]
fn k3_finish_fused_into_the_head_matches_the_k8_pass() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let k8 = K8Pipeline::new(&ctx.device).expect("k8 pipeline");
    let (inn, out) = (256usize, 96usize);
    let weights = k3
        .upload(ctx, "finish head", &q5k_bytes(61, inn, out), inn, out)
        .expect("upload head");
    let input = lcg(62, inn);
    let in_buf = ctx.upload_f32("finish input", &input);
    let ref_buf = ctx.scratch_buffer("finish ref", (out * 4) as u64);
    let new_buf = ctx.scratch_buffer("finish new", (out * 4) as u64);
    let (scale, cap) = (2.0f32, 30.0f32);
    let plain = k3
        .prepare(ctx, &weights, &in_buf, &ref_buf, 1)
        .expect("prepare plain head");
    let finish = k8
        .prepare_finish(ctx, &ref_buf, out, scale, cap)
        .expect("prepare k8 finish");
    let fused = k3
        .prepare_with_finish(
            ctx,
            &weights,
            &in_buf,
            &new_buf,
            1,
            FINISH_SOFTCAP,
            scale,
            cap,
        )
        .expect("prepare fused head");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        plain.encode_into(&mut pass);
        finish.encode_into(&mut pass);
        fused.encode_into(&mut pass);
    }
    ctx.queue.submit([encoder.finish()]);
    let want = ctx.read_f32(&ref_buf, out).expect("read ref");
    let got = ctx.read_f32(&new_buf, out).expect("read new");
    let d = max_abs(&got, &want);
    eprintln!("k3 fused finish (softcap): max abs diff {d:.3e}");
    assert!(d < 1e-5, "fused finish drifted: {d}");

    // The scale-only mode against K8's scale pass.
    let finish2 = k8.prepare_scale(ctx, &ref_buf, out, scale);
    let fused2 = k3
        .prepare_with_finish(
            ctx,
            &weights,
            &in_buf,
            &new_buf,
            1,
            FINISH_SCALE,
            scale,
            0.0,
        )
        .expect("prepare scale head");
    let plain2 = k3
        .prepare(ctx, &weights, &in_buf, &ref_buf, 1)
        .expect("prepare plain head 2");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        plain2.encode_into(&mut pass);
        finish2.encode_into(&mut pass);
        fused2.encode_into(&mut pass);
    }
    ctx.queue.submit([encoder.finish()]);
    let want = ctx.read_f32(&ref_buf, out).expect("read ref 2");
    let got = ctx.read_f32(&new_buf, out).expect("read new 2");
    let d = max_abs(&got, &want);
    eprintln!("k3 fused finish (scale): max abs diff {d:.3e}");
    assert!(d < 1e-5, "fused scale finish drifted: {d}");
}

#[test]
fn k8_gather_norm_matches_gather_then_plain_norm() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let k8 = K8Pipeline::new(&ctx.device).expect("k8 pipeline");
    let k4 = K4Pipeline::new(&ctx.device).expect("k4 pipeline");
    // The quantized family (the real model's embedding is
    // Q4_K): 5 table rows of width 512, two gathered.
    {
        let (vocab, width, rows) = (5usize, 512usize, 2usize);
        let bytes = q4k_bytes(71, width, vocab);
        let table = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gather norm table"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&table, 0, &bytes);
        let tokens = ctx.scratch_buffer("gather norm tokens", 8);
        ctx.queue
            .write_buffer(&tokens, 0, &[3, 0, 0, 0, 0, 0, 0, 0]);
        let tmp = ctx.scratch_buffer("gather norm tmp", (rows * width * 4) as u64);
        let ref_buf = ctx.scratch_buffer("gather norm ref", (rows * width * 4) as u64);
        let new_buf = ctx.scratch_buffer("gather norm new", (rows * width * 4) as u64);
        let gather = k8
            .prepare_gather_rows(ctx, &table, &tokens, &tmp, rows, width, Dtype::Q4K)
            .expect("prepare gather");
        let norm = k4.prepare_rms_norm_plain(ctx, &tmp, &ref_buf, rows, width, 1e-5);
        let fused = k8
            .prepare_gather_rows_norm(
                ctx,
                &table,
                &tokens,
                &new_buf,
                rows,
                width,
                1e-5,
                Dtype::Q4K,
            )
            .expect("prepare gather+norm");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            gather.encode_into(&mut pass);
            norm.encode_into(&mut pass);
            fused.encode_into(&mut pass);
        }
        ctx.queue.submit([encoder.finish()]);
        let want = ctx.read_f32(&ref_buf, rows * width).expect("read ref");
        let got = ctx.read_f32(&new_buf, rows * width).expect("read new");
        let d = max_abs(&got, &want);
        eprintln!("k8 gather+norm (Q4_K): max abs diff {d:.3e}");
        assert!(d < 1e-6, "gather+norm drifted: {d}");
    }
    // The F32 family at an awkward width.
    {
        let (vocab, width, rows) = (6usize, 300usize, 3usize);
        let table_data = lcg(72, vocab * width);
        let table = ctx.upload_f32("gather norm f32 table", &table_data);
        let tokens = ctx.scratch_buffer("gather norm tokens f32", 12);
        ctx.queue
            .write_buffer(&tokens, 0, &[2, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0]);
        let tmp = ctx.scratch_buffer("gather norm tmp f32", (rows * width * 4) as u64);
        let ref_buf = ctx.scratch_buffer("gather norm ref f32", (rows * width * 4) as u64);
        let new_buf = ctx.scratch_buffer("gather norm new f32", (rows * width * 4) as u64);
        let gather = k8
            .prepare_gather_rows(ctx, &table, &tokens, &tmp, rows, width, Dtype::F32)
            .expect("prepare gather f32");
        let norm = k4.prepare_rms_norm_plain(ctx, &tmp, &ref_buf, rows, width, 1e-5);
        let fused = k8
            .prepare_gather_rows_norm(
                ctx,
                &table,
                &tokens,
                &new_buf,
                rows,
                width,
                1e-5,
                Dtype::F32,
            )
            .expect("prepare gather+norm f32");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            gather.encode_into(&mut pass);
            norm.encode_into(&mut pass);
            fused.encode_into(&mut pass);
        }
        ctx.queue.submit([encoder.finish()]);
        let want = ctx.read_f32(&ref_buf, rows * width).expect("read ref f32");
        let got = ctx.read_f32(&new_buf, rows * width).expect("read new f32");
        let d = max_abs(&got, &want);
        eprintln!("k8 gather+norm (F32): max abs diff {d:.3e}");
        assert!(d < 1e-6, "gather+norm (F32) drifted: {d}");
    }
}

/// K7's branch-stable sigmoid, on the host.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// K7's silu, on the host.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Submit one prepared call and read its output buffer back.
fn run_call(
    gpu: &WgpuCompute,
    encode: impl FnOnce(&mut wgpu::CommandEncoder),
    out: &wgpu::Buffer,
    n: usize,
) -> Vec<f32> {
    let ctx = gpu.context();
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    ctx.read_f32(out, n).expect("read back")
}

#[test]
fn k1_upload_multi_matches_the_separate_uploads() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let inn = 512;
    let a = q4k_bytes(7, inn, 8);
    let b = q4k_bytes(8, inn, 4);
    let wa = k1.upload(ctx, "a", &a, inn, 8).expect("upload a");
    let wb = k1.upload(ctx, "b", &b, inn, 4).expect("upload b");
    let multi = k1
        .upload_multi(ctx, "ab", &[(&a, inn, 8), (&b, inn, 4)])
        .expect("upload multi");
    assert_eq!(multi.out_rows, 12);
    assert_eq!(multi.byte_len, wa.byte_len + wb.byte_len);
    let input = lcg(9, 2 * inn);
    let out_a = k1.run(ctx, &wa, &input).expect("run a");
    let out_b = k1.run(ctx, &wb, &input).expect("run b");
    let out_m = k1.run(ctx, &multi, &input).expect("run multi");
    // Batch rows are independent: row r of the concatenated
    // run is row r of a's run (rows 0..8) then b's (8..12),
    // per batch row, bit for bit (same kernel, same rows).
    for batch in 0..2 {
        let want: Vec<f32> = out_a[batch * 8..batch * 8 + 8]
            .iter()
            .chain(out_b[batch * 4..batch * 4 + 4].iter())
            .copied()
            .collect();
        let got = &out_m[batch * 12..batch * 12 + 12];
        assert_eq!(got, want.as_slice(), "batch row {batch} differs");
    }
}

#[test]
fn k1_gated_activation_matches_the_composed_gate() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let plain = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let gated =
        K1Pipeline::with_shape_act(&ctx.device, k1_shipped(ctx.device.features()), ACT_GATED)
            .expect("k1 gated pipeline");
    let inn = 512;
    let out_rows = 8;
    let batch = 2usize;
    let w = plain
        .upload(ctx, "o", &q4k_bytes(11, inn, out_rows), inn, out_rows)
        .expect("upload");
    let attn = lcg(12, batch * inn);
    let gate = lcg(13, batch * inn);
    // Reference, the unfused order in the forward: the
    // attention rows and gate rows both exist, K7 multiplies
    // them, THEN the projection runs over the product. Gate
    // the rows on the host with K7's math and run the plain
    // GEMV over the product.
    let gated_rows: Vec<f32> = attn
        .iter()
        .zip(gate.iter())
        .map(|(a, g)| a * sigmoid(*g))
        .collect();
    let want = plain.run(ctx, &w, &gated_rows).expect("composed run");

    // Fused, separate gate buffer.
    let a_buf = ctx.upload_f32("attn", &attn);
    let b_buf = ctx.upload_f32("gate", &gate);
    let out_buf = ctx.scratch_buffer("gated out", (batch * out_rows * 4) as u64);
    let call = gated
        .prepare_act(
            ctx,
            &w,
            &a_buf,
            &b_buf,
            &out_buf,
            batch as u32,
            &K1ActGeom {
                a_stride_elems: inn,
                b_stride_elems: inn,
                b_off_elems: 0,
            },
        )
        .expect("prepare gated");
    let got = run_call(&gpu, |e| call.encode(e), &out_buf, batch * out_rows);
    let d = max_abs(&got, &want);
    eprintln!("k1 gated (separate): max abs diff {d:.3e}");
    assert!(d < 1e-4, "gated fusion drifted: {d}");

    // Fused, gate segment inside a wide buffer (the batched
    // QKV layout's form): stride twice the width, the gate in
    // the second half, garbage in the first.
    let mut wide = lcg(14, batch * 2 * inn);
    for r in 0..batch {
        wide[r * 2 * inn + inn..r * 2 * inn + 2 * inn]
            .copy_from_slice(&gate[r * inn..r * inn + inn]);
    }
    let wide_buf = ctx.upload_f32("wide", &wide);
    let call = gated
        .prepare_act(
            ctx,
            &w,
            &a_buf,
            &wide_buf,
            &out_buf,
            batch as u32,
            &K1ActGeom {
                a_stride_elems: inn,
                b_stride_elems: 2 * inn,
                b_off_elems: inn,
            },
        )
        .expect("prepare gated wide");
    let got = run_call(&gpu, |e| call.encode(e), &out_buf, batch * out_rows);
    let d = max_abs(&got, &want);
    eprintln!("k1 gated (wide segment): max abs diff {d:.3e}");
    assert!(d < 1e-4, "gated fusion (wide) drifted: {d}");
}

#[test]
fn k1_and_k2_swiglu_activation_match_the_composed_ffn() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let inn = 512;
    let out_rows = 8;
    let batch = 2usize;
    let gate = lcg(21, batch * inn);
    let up = lcg(22, batch * inn);
    let act: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect();
    // The pair layout both fused forms read: per batch row,
    // the gate segment then the up segment in one buffer.
    let mut pair = Vec::with_capacity(batch * 2 * inn);
    for r in 0..batch {
        pair.extend_from_slice(&gate[r * inn..(r + 1) * inn]);
        pair.extend_from_slice(&up[r * inn..(r + 1) * inn]);
    }
    let pair_buf = ctx.upload_f32("pair", &pair);
    let geom = K1ActGeom {
        a_stride_elems: 2 * inn,
        b_stride_elems: 2 * inn,
        b_off_elems: inn,
    };

    // K1 (Q4_K down).
    let plain1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let swiglu1 =
        K1Pipeline::with_shape_act(&ctx.device, k1_shipped(ctx.device.features()), ACT_SWIGLU)
            .expect("k1 swiglu pipeline");
    let w1 = plain1
        .upload(ctx, "down4", &q4k_bytes(23, inn, out_rows), inn, out_rows)
        .expect("upload q4k down");
    let want1 = plain1.run(ctx, &w1, &act).expect("composed q4k run");
    let out_buf = ctx.scratch_buffer("swiglu out", (batch * out_rows * 4) as u64);
    let call = swiglu1
        .prepare_act(
            ctx,
            &w1,
            &pair_buf,
            &pair_buf,
            &out_buf,
            batch as u32,
            &geom,
        )
        .expect("prepare k1 swiglu");
    let got1 = run_call(&gpu, |e| call.encode(e), &out_buf, batch * out_rows);
    let d = max_abs(&got1, &want1);
    eprintln!("k1 swiglu (pair): max abs diff {d:.3e}");
    assert!(d < 1e-4, "k1 swiglu fusion drifted: {d}");

    // K2 (Q6_K down), the real model's down family.
    let plain2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let swiglu2 =
        K2Pipeline::with_shape_act(&ctx.device, k2_shipped(ctx.device.features()), ACT_SWIGLU)
            .expect("k2 swiglu pipeline");
    let w2 = plain2
        .upload(ctx, "down6", &q6k_bytes(24, inn, out_rows), inn, out_rows)
        .expect("upload q6k down");
    let want2 = plain2.run(ctx, &w2, &act).expect("composed q6k run");
    let call = swiglu2
        .prepare_act(
            ctx,
            &w2,
            &pair_buf,
            &pair_buf,
            &out_buf,
            batch as u32,
            &geom,
        )
        .expect("prepare k2 swiglu");
    let got2 = run_call(&gpu, |e| call.encode(e), &out_buf, batch * out_rows);
    let d = max_abs(&got2, &want2);
    eprintln!("k2 swiglu (pair): max abs diff {d:.3e}");
    assert!(d < 1e-3, "k2 swiglu fusion drifted: {d}");
}
