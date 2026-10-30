//! The K1 WGSL must compile and the pipeline must build on
//! whatever adapter a machine offers, including a software one:
//! shader validity is a property of the shader, not of the
//! selection policy (which exists to decide what may SERVE, and
//! rightly refuses llvmpipe for that). This test bypasses the
//! policy on purpose and uses the first adapter wgpu enumerates;
//! on a machine with no adapter at all it reports and passes.

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
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute shader compile test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device for shader compile test");
    let _pipeline = wgpu_compute::K1Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K1 pipeline must build on {}: {e}", info.name));
    eprintln!("K1 shader compiled on {} ({:?})", info.name, info.backend);
}
