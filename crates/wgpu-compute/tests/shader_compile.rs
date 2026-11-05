//! The K1 WGSL must compile and the pipelines must build on
//! whatever adapter a machine offers, including a software one:
//! shader validity is a property of the shader, not of the
//! selection policy (which exists to decide what may SERVE, and
//! rightly refuses llvmpipe for that). This test bypasses the
//! policy on purpose and uses the first adapter wgpu enumerates;
//! on a machine with no adapter at all it reports and passes.
//!
//! Every [`K1Shape`] the test device's granted features allow is
//! compiled: the device here is opened with the features the
//! shapes need (the same mask `GpuContext` uses; asking for
//! every feature an adapter exposes fails on stacks that list
//! experimental features wgpu will not enable), so on the
//! server stack the subgroup and f16 variants are validated
//! too. A shape the device cannot run must refuse by name in
//! `with_shape`, which is also asserted.

use wgpu_compute::kernels::k1_q4k_gemv::K1Shape;
use wgpu_compute::kernels::k2_q6k_gemv::K2Shape;
use wgpu_compute::kernels::k3_q5k_gemv::K3Shape;
use wgpu_compute::{K1Pipeline, K2Pipeline, K3Pipeline};

#[test]
fn k1_shader_compiles_on_any_available_adapter() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping shader compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute shader compile test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device for shader compile test");
    let _pipeline = K1Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K1 pipeline must build on {}: {e}", info.name));
    eprintln!(
        "K1 shipped shader compiled on {} ({:?})",
        info.name, info.backend
    );

    let shapes = [
        K1Shape::LEGACY,
        K1Shape::llama(1, 128),
        K1Shape::llama(2, 128),
        K1Shape::llama(4, 128),
        K1Shape::llama(2, 64),
        K1Shape::llama(4, 256),
        K1Shape::llama(2, 128).with_subgroup(),
        K1Shape::llama(4, 128).with_subgroup(),
        K1Shape::llama(2, 128).with_f16(),
        K1Shape::llama(4, 128).with_subgroup().with_f16(),
    ];
    let features = device.features();
    for shape in shapes {
        match K1Pipeline::with_shape(&device, shape) {
            Ok(_) => eprintln!("shape {} compiled on {}", shape.name(), info.name),
            Err(e) => {
                // A refusal is only acceptable when it names the
                // feature this device was not granted.
                let lacks = (shape.subgroup && !features.contains(wgpu::Features::SUBGROUP))
                    || (shape.f16_activations && !features.contains(wgpu::Features::SHADER_F16));
                assert!(
                    lacks,
                    "shape {} must build on {} with features {features:?}: {e}",
                    shape.name(),
                    info.name
                );
                eprintln!("shape {} refused as expected: {e}", shape.name());
            }
        }
    }
}

#[test]
fn k2_k3_shaders_compile_on_any_available_adapter() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping K2/K3 shader compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute K2/K3 shader compile test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device for K2/K3 shader compile test");
    let _k2 = K2Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K2 pipeline must build on {}: {e}", info.name));
    let _k3 = K3Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K3 pipeline must build on {}: {e}", info.name));
    eprintln!(
        "K2 and K3 shipped shaders compiled on {} ({:?})",
        info.name, info.backend
    );

    let features = device.features();
    let k2_shapes = [
        K2Shape::llama(1, 128),
        K2Shape::llama(2, 128),
        K2Shape::llama(4, 128),
        K2Shape::llama(2, 64),
        K2Shape::llama(2, 128).with_subgroup(),
        K2Shape::llama(4, 128).with_subgroup(),
    ];
    for shape in k2_shapes {
        match K2Pipeline::with_shape(&device, shape) {
            Ok(_) => eprintln!("K2 shape {} compiled on {}", shape.name(), info.name),
            Err(e) => {
                let lacks = shape.subgroup && !features.contains(wgpu::Features::SUBGROUP);
                assert!(
                    lacks,
                    "K2 shape {} must build on {} with features {features:?}: {e}",
                    shape.name(),
                    info.name
                );
                eprintln!("K2 shape {} refused as expected: {e}", shape.name());
            }
        }
    }
    let k3_shapes = [
        K3Shape::llama(1, 128),
        K3Shape::llama(2, 128),
        K3Shape::llama(4, 128),
        K3Shape::llama(2, 64),
        K3Shape::llama(2, 128).with_subgroup(),
        K3Shape::llama(4, 128).with_subgroup(),
    ];
    for shape in k3_shapes {
        match K3Pipeline::with_shape(&device, shape) {
            Ok(_) => eprintln!("K3 shape {} compiled on {}", shape.name(), info.name),
            Err(e) => {
                let lacks = shape.subgroup && !features.contains(wgpu::Features::SUBGROUP);
                assert!(
                    lacks,
                    "K3 shape {} must build on {} with features {features:?}: {e}",
                    shape.name(),
                    info.name
                );
                eprintln!("K3 shape {} refused as expected: {e}", shape.name());
            }
        }
    }
}

#[test]
fn k12_k14_mmq_shaders_compile_on_any_available_adapter() {
    use wgpu_compute::kernels::k12_q8_quant::K12QuantPipeline;
    use wgpu_compute::kernels::k13_q4k_mmq::{K13Pipeline, MmqShape};
    use wgpu_compute::kernels::k14_q6k_mmq::K14Pipeline;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping k12-k14 compile test: wgpu enumerates no adapters here");
        return;
    };
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    // First device: wgpu's default limits (16 KiB of
    // workgroup storage).
    let (capped, _q1) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute k12-k14 capped test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("capped device for k12-k14 test");
    // The K12 quantizer builds under any granted storage
    // (its only workgroup memory is the 2 KB reduction
    // tree), in all three transform forms.
    for act in [
        wgpu_compute::kernels::k1_q4k_gemv::ACT_NONE,
        wgpu_compute::kernels::k1_q4k_gemv::ACT_GATED,
        wgpu_compute::kernels::k1_q4k_gemv::ACT_SWIGLU,
    ] {
        K12QuantPipeline::new(&capped, act).expect("k12 quantizer shader must compile");
    }
    // Under the default limits the MMQ staging does not
    // fit, and the constructors must refuse that by name,
    // not panic or silently degrade.
    match K13Pipeline::for_device(&capped) {
        Ok(None) => {}
        Ok(Some(_)) => panic!("k13 under default limits must be Ok(None), got a pipeline"),
        Err(e) => panic!("k13 under default limits must be Ok(None), got error: {e}"),
    }
    match K14Pipeline::for_device(&capped) {
        Ok(None) => {}
        Ok(Some(_)) => panic!("k14 under default limits must be Ok(None), got a pipeline"),
        Err(e) => panic!("k14 under default limits must be Ok(None), got error: {e}"),
    }
    let err = match K13Pipeline::with_shape(&capped, MmqShape::new(32)) {
        Ok(_) => panic!("k13 m32 must refuse a 16 KiB storage grant by name"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("workgroup memory"),
        "refusal names the binding limit: {err}"
    );
    // Second device: the adapter's own limits. Where they
    // grant the staging (the server's card), both MMQ
    // shaders compile at their shipped shapes.
    let (full, _q2) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute k12-k14 full test"),
        required_features: wanted,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("full-limits device for k12-k14 test");
    if let Ok(Some(_)) = K13Pipeline::for_device(&full) {
        eprintln!("k13: shipped shape built on this adapter");
    }
    if let Ok(Some(_)) = K14Pipeline::for_device(&full) {
        eprintln!("k14: shipped shape built on this adapter");
    }
}

#[test]
fn k4_to_k8_pipelines_compile_on_any_available_adapter() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping K4-K8 compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute K4-K8 compile test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device for K4-K8 compile test");
    // K8 in particular: the multi-row gathers and the
    // scale-only finish landed with the resident session, and
    // this is their adapter-independent compile gate.
    let _k4 = wgpu_compute::K4Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K4 pipeline must build on {}: {e}", info.name));
    let _k5 = wgpu_compute::K5Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K5 pipeline must build on {}: {e}", info.name));
    let _k6 = wgpu_compute::K6Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K6 pipeline must build on {}: {e}", info.name));
    let _k7 = wgpu_compute::K7Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K7 pipeline must build on {}: {e}", info.name));
    let _k8 = wgpu_compute::K8Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K8 pipeline must build on {}: {e}", info.name));
    eprintln!(
        "K4-K8 shipped shaders compiled on {} ({:?})",
        info.name, info.backend
    );
}

#[test]
fn k10_k11_gemm_shaders_compile_on_any_available_adapter() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping K10/K11 shader compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute K10/K11 shader compile test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device for K10/K11 shader compile test");
    use wgpu_compute::kernels::k10_q4k_gemm::{GemmShape, K10Pipeline};
    use wgpu_compute::kernels::k11_q6k_gemm::K11Pipeline;
    use wgpu_compute::kernels::k1_q4k_gemv::{ACT_GATED, ACT_NONE, ACT_SWIGLU};
    // The narrow tile fits the default limits' 16 KiB of
    // workgroup storage, so it must build everywhere; the
    // device-selected shapes go through for_device, which
    // steps down or reports absence by the same limits.
    let narrow = GemmShape::new(32, 128);
    for act in [ACT_NONE, ACT_GATED, ACT_SWIGLU] {
        K10Pipeline::with_shape_act(&device, narrow, act)
            .unwrap_or_else(|e| panic!("K10 narrow act {act} must build on {}: {e}", info.name));
    }
    for act in [ACT_NONE, ACT_SWIGLU] {
        K11Pipeline::with_shape_act(&device, narrow, act)
            .unwrap_or_else(|e| panic!("K11 narrow act {act} must build on {}: {e}", info.name));
    }
    let _k10 = K10Pipeline::for_device(&device, ACT_NONE)
        .unwrap_or_else(|e| panic!("K10 for_device must not fail on {}: {e}", info.name));
    let _k11 = K11Pipeline::for_device(&device, ACT_NONE)
        .unwrap_or_else(|e| panic!("K11 for_device must not fail on {}: {e}", info.name));
    eprintln!(
        "K10 and K11 GEMM shaders compiled on {} ({:?})",
        info.name, info.backend
    );
}

#[test]
fn k17_dflash_shaders_compile_on_any_available_adapter() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping K17 shader compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute K17 shader compile test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device for K17 shader compile test");
    let _pipeline = wgpu_compute::K17Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K17 pipelines must build on {}: {e}", info.name));
    eprintln!(
        "K17 shipped shaders compiled on {} ({:?})",
        info.name, info.backend
    );
}

#[test]
fn k15_k16_mmq_wide_shaders_compile_on_any_available_adapter() {
    use wgpu_compute::kernels::k15_q4k_mmq_wide::{
        q4k_wide_staging_bytes, shipped_wide_shape as shipped_q4k, K15Pipeline, WideMmqShape,
        WIDE_TILE_CANDIDATES,
    };
    use wgpu_compute::kernels::k16_q6k_mmq_wide::{
        q6k_wide_staging_bytes, shipped_wide_shape as shipped_q6k, K16Pipeline,
    };
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping k15-k16 compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    // First device: wgpu's default limits (16 KiB of
    // workgroup storage). No wide candidate's
    // double-buffered staging set fits there, so both
    // constructors must report absence (for_device) or
    // refuse by name (with_shape), never panic or degrade.
    let (capped, _q1) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute k15-k16 capped test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("capped device for k15-k16 test");
    match K15Pipeline::for_device(&capped) {
        Ok(None) => {}
        Ok(Some(_)) => panic!("k15 under default limits must be Ok(None), got a pipeline"),
        Err(e) => panic!("k15 under default limits must be Ok(None), got error: {e}"),
    }
    match K16Pipeline::for_device(&capped) {
        Ok(None) => {}
        Ok(Some(_)) => panic!("k16 under default limits must be Ok(None), got a pipeline"),
        Err(e) => panic!("k16 under default limits must be Ok(None), got error: {e}"),
    }
    let err = match K15Pipeline::with_shape(&capped, WideMmqShape::new(64)) {
        Ok(_) => panic!("k15 w64 must refuse a 16 KiB storage grant by name"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("workgroup memory"),
        "refusal names the binding limit: {err}"
    );
    // A tile width outside the candidate set refuses by
    // name on any device.
    let err = match K15Pipeline::with_shape(&capped, WideMmqShape::new(100)) {
        Ok(_) => panic!("k15 must refuse a non-candidate tile width by name"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("is not one of"),
        "refusal names the candidate set: {err}"
    );
    // Second device: the adapter's own limits. Every
    // candidate whose staging the grant covers must
    // compile: this is the leg that validates the wide
    // shaders (the parity-unrolled pipeline, the parity
    // buffer sets) on whatever stack runs the test,
    // including this repo's build VM.
    let (full, _q2) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute k15-k16 full test"),
        required_features: wanted,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("full-limits device for k15-k16 test");
    let granted = full.limits().max_compute_workgroup_storage_size;
    for cand in WIDE_TILE_CANDIDATES {
        let shape = WideMmqShape::new(cand);
        if granted >= q4k_wide_staging_bytes(&shape) {
            K15Pipeline::with_shape(&full, shape).unwrap_or_else(|e| {
                panic!(
                    "k15 {} must build on {} (grant {granted}): {e}",
                    shape.name(),
                    info.name
                )
            });
            eprintln!("k15 {} compiled on {}", shape.name(), info.name);
        }
        if granted >= q6k_wide_staging_bytes(&shape) {
            K16Pipeline::with_shape(&full, shape).unwrap_or_else(|e| {
                panic!(
                    "k16 {} must build on {} (grant {granted}): {e}",
                    shape.name(),
                    info.name
                )
            });
            eprintln!("k16 {} compiled on {}", shape.name(), info.name);
        }
    }
    // for_device picks exactly the shipped shape the
    // limits arithmetic names, on both families.
    match (shipped_q4k(&full.limits()), K15Pipeline::for_device(&full)) {
        (Some(want), Ok(Some(p))) => assert_eq!(p.shape(), want),
        (None, Ok(None)) => {}
        (want, got) => panic!(
            "k15 for_device disagrees with the shipped shape: want {want:?}, got {:?}",
            got.map(|o| o.map(|p| p.shape()))
        ),
    }
    match (shipped_q6k(&full.limits()), K16Pipeline::for_device(&full)) {
        (Some(want), Ok(Some(p))) => assert_eq!(p.shape(), want),
        (None, Ok(None)) => {}
        (want, got) => panic!(
            "k16 for_device disagrees with the shipped shape: want {want:?}, got {:?}",
            got.map(|o| o.map(|p| p.shape()))
        ),
    }
}

#[test]
fn k18_small_batch_shaders_compile_on_any_available_adapter() {
    use wgpu_compute::kernels::k18_small_batch::{
        q4k_small_staging_bytes, q5k_small_staging_bytes, q6k_small_staging_bytes,
        shipped_shape_q4k, shipped_shape_q5k, shipped_shape_q6k, K18Q4KPipeline, K18Q5KPipeline,
        K18Q6KPipeline, SmallShape,
    };
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping k18 compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    // First device: wgpu's default limits (16 KiB of
    // workgroup storage). The Q4_K and Q5_K tile-32
    // staging sets fit that grant (15,424 bytes), so
    // their constructors must produce the s32 pipeline;
    // the Q6_K set is 64 bytes over it, so its
    // constructor must report absence. The wide tile
    // refuses by name on all three, and a tile width
    // outside the candidate set refuses by name on any
    // device.
    let (capped, _q1) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute k18 capped test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("capped device for k18 test");
    match K18Q4KPipeline::for_device(&capped) {
        Ok(Some(p)) => assert_eq!(p.shape(), SmallShape::new(32)),
        other => panic!("k18 q4k under default limits must be Ok(Some(s32)), got {other:?}"),
    }
    match K18Q5KPipeline::for_device(&capped) {
        Ok(Some(p)) => assert_eq!(p.shape(), SmallShape::new(32)),
        other => panic!("k18 q5k under default limits must be Ok(Some(s32)), got {other:?}"),
    }
    match K18Q6KPipeline::for_device(&capped) {
        Ok(None) => {}
        Ok(Some(p)) => panic!(
            "k18 q6k under default limits must be Ok(None), got a pipeline at {:?}",
            p.shape()
        ),
        Err(e) => panic!("k18 q6k under default limits must be Ok(None), got error: {e}"),
    }
    let err = match K18Q4KPipeline::with_shape(&capped, SmallShape::new(64)) {
        Ok(_) => panic!("k18 q4k s64 must refuse a 16 KiB storage grant by name"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("workgroup memory"),
        "refusal names the binding limit: {err}"
    );
    let err = match K18Q4KPipeline::with_shape(&capped, SmallShape::new(100)) {
        Ok(_) => panic!("k18 must refuse a non-candidate tile width by name"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("not 32 or 64"),
        "refusal names the candidate set: {err}"
    );
    // Second device: the adapter's own limits. Every
    // candidate whose staging the grant covers must
    // compile, on all three families: this is the leg
    // that validates the small-batch shaders (the
    // four-batch-lane setup, the Q5_K assembly stage)
    // on whatever stack runs the test.
    let (full, _q2) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute k18 full test"),
        required_features: wanted,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("full-limits device for k18 test");
    let granted = full.limits().max_compute_workgroup_storage_size;
    for cand in [64, 32] {
        let shape = SmallShape::new(cand);
        if granted >= q4k_small_staging_bytes(&shape) {
            K18Q4KPipeline::with_shape(&full, shape).unwrap_or_else(|e| {
                panic!(
                    "k18 q4k {} must build on {} (grant {granted}): {e}",
                    shape.name(),
                    info.name
                )
            });
            eprintln!("k18 q4k {} compiled on {}", shape.name(), info.name);
        }
        if granted >= q5k_small_staging_bytes(&shape) {
            K18Q5KPipeline::with_shape(&full, shape).unwrap_or_else(|e| {
                panic!(
                    "k18 q5k {} must build on {} (grant {granted}): {e}",
                    shape.name(),
                    info.name
                )
            });
            eprintln!("k18 q5k {} compiled on {}", shape.name(), info.name);
        }
        if granted >= q6k_small_staging_bytes(&shape) {
            K18Q6KPipeline::with_shape(&full, shape).unwrap_or_else(|e| {
                panic!(
                    "k18 q6k {} must build on {} (grant {granted}): {e}",
                    shape.name(),
                    info.name
                )
            });
            eprintln!("k18 q6k {} compiled on {}", shape.name(), info.name);
        }
    }
    // for_device picks exactly the shipped shape the
    // limits arithmetic names, on all three families.
    match (
        shipped_shape_q4k(&full.limits()),
        K18Q4KPipeline::for_device(&full),
    ) {
        (Some(want), Ok(Some(p))) => assert_eq!(p.shape(), want),
        (None, Ok(None)) => {}
        (want, got) => panic!(
            "k18 q4k for_device disagrees with the shipped shape: want {want:?}, got {:?}",
            got.map(|o| o.map(|p| p.shape()))
        ),
    }
    match (
        shipped_shape_q5k(&full.limits()),
        K18Q5KPipeline::for_device(&full),
    ) {
        (Some(want), Ok(Some(p))) => assert_eq!(p.shape(), want),
        (None, Ok(None)) => {}
        (want, got) => panic!(
            "k18 q5k for_device disagrees with the shipped shape: want {want:?}, got {:?}",
            got.map(|o| o.map(|p| p.shape()))
        ),
    }
    match (
        shipped_shape_q6k(&full.limits()),
        K18Q6KPipeline::for_device(&full),
    ) {
        (Some(want), Ok(Some(p))) => assert_eq!(p.shape(), want),
        (None, Ok(None)) => {}
        (want, got) => panic!(
            "k18 q6k for_device disagrees with the shipped shape: want {want:?}, got {:?}",
            got.map(|o| o.map(|p| p.shape()))
        ),
    }
}

#[test]
fn k19_mmv_shaders_compile_on_any_available_adapter() {
    use wgpu_compute::kernels::k19_q4k_mmv::{MmvShape, MMV_MAX_BATCH, MMV_MIN_BATCH};
    use wgpu_compute::K19Pipeline;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping k19 compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let wanted = adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16);
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute k19 compile test"),
        required_features: wanted,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device for k19 compile test");
    let features = device.features();
    // The shipped pipeline builds on any adapter: its
    // only workgroup memory is the reduction partials,
    // far under the default grant.
    let _pipeline = K19Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K19 pipeline must build on {}: {e}", info.name));
    eprintln!(
        "K19 shipped shader compiled on {} ({:?})",
        info.name, info.backend
    );
    // Every batch variant in the band compiles, for
    // every shape the device can run: the batch is a
    // generation literal, so each count is its own
    // module and each one is gated here, not just the
    // band floor the constructor compiles.
    for shape in [MmvShape::new(2, 128), MmvShape::new(2, 128).with_subgroup()] {
        let pipe = match K19Pipeline::with_shape(&device, shape) {
            Ok(p) => p,
            Err(e) => {
                // A refusal is only acceptable when it
                // names the feature this device was not
                // granted.
                assert!(
                    shape.subgroup && !features.contains(wgpu::Features::SUBGROUP),
                    "shape {} must build on {} with features {features:?}: {e}",
                    shape.name(),
                    info.name
                );
                eprintln!("shape {} refused as expected: {e}", shape.name());
                continue;
            }
        };
        for batch in MMV_MIN_BATCH..=MMV_MAX_BATCH {
            pipe.precompile(&device, batch).unwrap_or_else(|e| {
                panic!(
                    "k19 shape {} batch {batch} must compile on {}: {e}",
                    shape.name(),
                    info.name
                )
            });
        }
        eprintln!(
            "k19 shape {} compiled all batches {MMV_MIN_BATCH}..={MMV_MAX_BATCH} on {}",
            shape.name(),
            info.name
        );
        // Past the cap the variant request refuses by
        // name.
        let err = pipe
            .precompile(&device, MMV_MAX_BATCH + 1)
            .expect_err("batch past the cap must refuse");
        assert!(
            err.to_string().contains("outside 2..=16"),
            "refusal names the band: {err}"
        );
    }
}
