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
use wgpu_compute::K1Pipeline;

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
