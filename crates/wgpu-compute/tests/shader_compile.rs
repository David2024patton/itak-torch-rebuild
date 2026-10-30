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
